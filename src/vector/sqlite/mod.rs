mod codec;
mod identity;
mod mutation;
mod snapshot;
mod storage;

use super::in_memory::{validate_partition, validate_request_filters};
use super::{
    VectorIndex, VectorIndexChangeToken, VectorIndexDescriptor, VectorIndexObservation,
    VectorIndexStatus, VectorMutationConsistency, VectorRecord, VectorResult, VectorRevision,
    VectorSearchRequest, VectorSearchResult,
};
use rusqlite::Connection;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Durable exact-vector index backed by one SQLite database.
///
/// All database access runs on Tokio's blocking pool. Independent handles and
/// processes coordinate mutations through SQLite `IMMEDIATE` transactions,
/// so revision-conditioned partition replacement has one global
/// linearization point. The synchronous status accessors are cached
/// compatibility views; use [`VectorIndex::observe`] for current durable
/// evidence. On Unix and Windows, copying or atomically replacing the closed
/// database forks its history token on the next open. In-place overwrite and
/// concurrent out-of-band file operations are not supported.
#[derive(Clone)]
pub struct SqliteVectorIndex {
    inner: Arc<IndexInner>,
}

struct IndexInner {
    descriptor: VectorIndexDescriptor,
    connection: Arc<Mutex<Connection>>,
    observation: RwLock<VectorIndexObservation>,
}

impl std::fmt::Debug for SqliteVectorIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteVectorIndex")
            .field("descriptor", &self.inner.descriptor)
            .field("status", &self.status())
            .finish_non_exhaustive()
    }
}

impl SqliteVectorIndex {
    /// Open or initialize a durable index at `path`.
    ///
    /// Reopening requires the exact descriptor used to create the database.
    /// Existing rows and accounting metadata are validated before the handle
    /// is returned.
    pub async fn open(
        path: impl AsRef<Path>,
        descriptor: VectorIndexDescriptor,
    ) -> VectorResult<Self> {
        descriptor.validate()?;
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent).await.map_err(|_| {
                super::VectorIndexError::StorageFailed(
                    "could not create the SQLite parent directory".to_string(),
                )
            })?;
        }
        let open_descriptor = descriptor.clone();
        let (connection, observation) =
            tokio::task::spawn_blocking(move || storage::open(&path, &open_descriptor))
                .await
                .map_err(|_| {
                    super::VectorIndexError::WorkerFailed(
                        "SQLite vector-index initialization did not complete".to_string(),
                    )
                })??;

        Ok(Self {
            inner: Arc::new(IndexInner {
                descriptor,
                connection: Arc::new(Mutex::new(connection)),
                observation: RwLock::new(observation),
            }),
        })
    }

    async fn with_connection<T, F>(&self, operation: F) -> VectorResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> VectorResult<T> + Send + 'static,
    {
        let connection = Arc::clone(&self.inner.connection);
        tokio::task::spawn_blocking(move || {
            let mut connection = connection.lock().map_err(|_| {
                super::VectorIndexError::StorageFailed(
                    "SQLite connection lock is unavailable".to_string(),
                )
            })?;
            operation(&mut connection)
        })
        .await
        .map_err(|_| {
            super::VectorIndexError::WorkerFailed(
                "SQLite vector-index operation did not complete".to_string(),
            )
        })?
    }

    fn cache_observation(&self, observation: VectorIndexObservation) {
        *write_unpoisoned(&self.inner.observation) = observation;
    }

    async fn finish_mutation<F>(&self, operation: F) -> VectorResult<VectorIndexStatus>
    where
        F: FnOnce(&mut Connection) -> VectorResult<mutation::MutationOutcome> + Send + 'static,
    {
        let outcome = self.with_connection(operation).await?;
        self.cache_observation(outcome.observation);
        outcome.result
    }
}

#[async_trait::async_trait]
impl VectorIndex for SqliteVectorIndex {
    fn descriptor(&self) -> &VectorIndexDescriptor {
        &self.inner.descriptor
    }

    fn status(&self) -> VectorIndexStatus {
        read_unpoisoned(&self.inner.observation).status.clone()
    }

    fn change_token(&self) -> Option<VectorIndexChangeToken> {
        read_unpoisoned(&self.inner.observation)
            .change_token
            .clone()
    }

    async fn observe(&self) -> VectorResult<VectorIndexObservation> {
        let descriptor = self.inner.descriptor.clone();
        let observation = self
            .with_connection(move |connection| storage::observe(connection, &descriptor))
            .await?;
        self.cache_observation(observation.clone());
        Ok(observation)
    }

    fn mutation_consistency(&self) -> VectorMutationConsistency {
        VectorMutationConsistency::IndexRevisionCas
    }

    async fn replace_partition(
        &self,
        partition: &str,
        records: Vec<VectorRecord>,
    ) -> VectorResult<VectorIndexStatus> {
        let partition = validate_partition(partition)?.to_string();
        let descriptor = self.inner.descriptor.clone();
        self.finish_mutation(move |connection| {
            mutation::replace_partition(connection, &descriptor, partition, records, None)
        })
        .await
    }

    async fn replace_partition_if_revision(
        &self,
        partition: &str,
        expected_revision: VectorRevision,
        records: Vec<VectorRecord>,
    ) -> VectorResult<VectorIndexStatus> {
        let partition = validate_partition(partition)?.to_string();
        let descriptor = self.inner.descriptor.clone();
        self.finish_mutation(move |connection| {
            mutation::replace_partition(
                connection,
                &descriptor,
                partition,
                records,
                Some(expected_revision),
            )
        })
        .await
    }

    async fn remove_partition(&self, partition: &str) -> VectorResult<VectorIndexStatus> {
        let partition = validate_partition(partition)?.to_string();
        let descriptor = self.inner.descriptor.clone();
        self.finish_mutation(move |connection| {
            mutation::remove_partition(connection, &descriptor, partition, None)
        })
        .await
    }

    async fn remove_partition_if_revision(
        &self,
        partition: &str,
        expected_revision: VectorRevision,
    ) -> VectorResult<VectorIndexStatus> {
        let partition = validate_partition(partition)?.to_string();
        let descriptor = self.inner.descriptor.clone();
        self.finish_mutation(move |connection| {
            mutation::remove_partition(connection, &descriptor, partition, Some(expected_revision))
        })
        .await
    }

    async fn search(&self, request: VectorSearchRequest) -> VectorResult<VectorSearchResult> {
        validate_request_filters(&request)?;
        if request.limit == 0 {
            return Err(super::VectorIndexError::InvalidRequest(
                "limit must be greater than zero".to_string(),
            ));
        }
        let descriptor = self.inner.descriptor.clone();
        let (result, observation) = self
            .with_connection(move |connection| storage::search(connection, &descriptor, request))
            .await?;
        self.cache_observation(observation);
        Ok(result)
    }

    async fn clear(&self) -> VectorResult<VectorIndexStatus> {
        let descriptor = self.inner.descriptor.clone();
        self.finish_mutation(move |connection| mutation::clear(connection, &descriptor))
            .await
    }
}

fn read_unpoisoned<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_unpoisoned<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
