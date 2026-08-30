use super::search::search_snapshot;
use super::{
    VectorBudgetResource, VectorIndex, VectorIndexChangeToken, VectorIndexDescriptor,
    VectorIndexError, VectorIndexObservation, VectorIndexStatus, VectorMutationConsistency,
    VectorNormalization, VectorRecord, VectorResult, VectorRevision, VectorSearchRequest,
    VectorSearchResult,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

const HISTORY_DIGEST_DOMAIN: &str = "a3s.memory.vector-index-history.v1";

/// Exact, session-ephemeral vector index backed by immutable partition blocks.
#[derive(Clone)]
pub struct InMemoryVectorIndex {
    inner: Arc<IndexInner>,
}

impl std::fmt::Debug for InMemoryVectorIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InMemoryVectorIndex")
            .field("descriptor", &self.inner.descriptor)
            .field("status", &self.status())
            .finish()
    }
}

struct IndexInner {
    descriptor: VectorIndexDescriptor,
    initial_change_token: VectorIndexChangeToken,
    snapshot: RwLock<Arc<IndexSnapshot>>,
}

#[derive(Default)]
pub(super) struct IndexSnapshot {
    pub(super) revision: VectorRevision,
    pub(super) partitions: BTreeMap<String, Arc<PartitionBlock>>,
    pub(super) record_count: usize,
    pub(super) byte_count: usize,
}

pub(super) struct PartitionBlock {
    pub(super) name: String,
    pub(super) ids: Vec<String>,
    pub(super) labels: Vec<BTreeMap<String, String>>,
    pub(super) vectors: Vec<f32>,
    pub(super) byte_count: usize,
}

impl PartitionBlock {
    pub(super) fn record_count(&self) -> usize {
        self.ids.len()
    }
}

impl IndexSnapshot {
    pub(super) fn status(&self) -> VectorIndexStatus {
        VectorIndexStatus {
            revision: self.revision,
            partition_count: self.partitions.len(),
            record_count: self.record_count,
            byte_count: self.byte_count,
        }
    }
}

impl InMemoryVectorIndex {
    pub fn new(descriptor: VectorIndexDescriptor) -> VectorResult<Self> {
        descriptor.validate()?;
        let initial_change_token =
            VectorIndexChangeToken::try_new(new_history_digest(), VectorRevision::default())?;
        Ok(Self {
            inner: Arc::new(IndexInner {
                descriptor,
                initial_change_token,
                snapshot: RwLock::new(Arc::new(IndexSnapshot::default())),
            }),
        })
    }

    fn snapshot(&self) -> Arc<IndexSnapshot> {
        read_unpoisoned(&self.inner.snapshot).clone()
    }
}

pub(super) fn new_history_digest() -> String {
    let mut hasher = Sha256::new();
    hasher.update(HISTORY_DIGEST_DOMAIN.as_bytes());
    hasher.update([0]);
    hasher.update(uuid::Uuid::new_v4().as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

#[async_trait::async_trait]
impl VectorIndex for InMemoryVectorIndex {
    fn descriptor(&self) -> &VectorIndexDescriptor {
        &self.inner.descriptor
    }

    fn status(&self) -> VectorIndexStatus {
        self.snapshot().status()
    }

    fn change_token(&self) -> Option<VectorIndexChangeToken> {
        let snapshot = self.snapshot();
        Some(
            self.inner
                .initial_change_token
                .with_revision(snapshot.revision),
        )
    }

    async fn observe(&self) -> VectorResult<VectorIndexObservation> {
        let snapshot = self.snapshot();
        let observation = VectorIndexObservation {
            status: snapshot.status(),
            change_token: Some(
                self.inner
                    .initial_change_token
                    .with_revision(snapshot.revision),
            ),
        };
        observation.verify()?;
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
        let inner = Arc::clone(&self.inner);
        run_blocking(move || {
            let block = build_partition(&inner.descriptor, partition, records)?;
            publish_partition(&inner, block, None)
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
        let inner = Arc::clone(&self.inner);
        run_blocking(move || {
            let block = build_partition(&inner.descriptor, partition, records)?;
            publish_partition(&inner, block, Some(expected_revision))
        })
        .await
    }

    async fn remove_partition(&self, partition: &str) -> VectorResult<VectorIndexStatus> {
        let partition = validate_partition(partition)?.to_string();
        let inner = Arc::clone(&self.inner);
        run_blocking(move || remove_partition(&inner, &partition, None)).await
    }

    async fn remove_partition_if_revision(
        &self,
        partition: &str,
        expected_revision: VectorRevision,
    ) -> VectorResult<VectorIndexStatus> {
        let partition = validate_partition(partition)?.to_string();
        let inner = Arc::clone(&self.inner);
        run_blocking(move || remove_partition(&inner, &partition, Some(expected_revision))).await
    }

    async fn search(&self, mut request: VectorSearchRequest) -> VectorResult<VectorSearchResult> {
        validate_request_filters(&request)?;
        if request.limit == 0 {
            return Err(VectorIndexError::InvalidRequest(
                "limit must be greater than zero".to_string(),
            ));
        }
        let query = prepare_vector(
            std::mem::take(&mut request.embedding),
            &self.inner.descriptor,
            "query".to_string(),
        )?;
        let descriptor = self.inner.descriptor.clone();
        let snapshot = self.snapshot();
        run_blocking(move || search_snapshot(snapshot, &descriptor, query, request)).await
    }

    async fn clear(&self) -> VectorResult<VectorIndexStatus> {
        let inner = Arc::clone(&self.inner);
        run_blocking(move || clear_index(&inner)).await
    }
}

async fn run_blocking<T, F>(operation: F) -> VectorResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> VectorResult<T> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| VectorIndexError::WorkerFailed(error.to_string()))?
}

pub(super) fn validate_partition(partition: &str) -> VectorResult<&str> {
    let partition = partition.trim();
    if partition.is_empty() {
        Err(VectorIndexError::InvalidPartition)
    } else {
        Ok(partition)
    }
}

pub(super) fn validate_request_filters(request: &VectorSearchRequest) -> VectorResult<()> {
    if request
        .partitions
        .iter()
        .any(|partition| partition.trim().is_empty())
    {
        return Err(VectorIndexError::InvalidPartition);
    }
    if request.labels.keys().any(|key| key.trim().is_empty()) {
        return Err(VectorIndexError::InvalidLabel {
            context: "query filter".to_string(),
        });
    }
    Ok(())
}

pub(super) fn build_partition(
    descriptor: &VectorIndexDescriptor,
    name: String,
    records: Vec<VectorRecord>,
) -> VectorResult<Arc<PartitionBlock>> {
    if records.len() > descriptor.max_records {
        return Err(VectorIndexError::BudgetExceeded {
            resource: VectorBudgetResource::Records,
            limit: descriptor.max_records,
            required: records.len(),
        });
    }
    let minimum_vector_bytes = records
        .len()
        .checked_mul(descriptor.dimension)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(VectorIndexError::SizeOverflow)?;
    if minimum_vector_bytes > descriptor.max_bytes {
        return Err(VectorIndexError::BudgetExceeded {
            resource: VectorBudgetResource::Bytes,
            limit: descriptor.max_bytes,
            required: minimum_vector_bytes,
        });
    }
    let mut seen = BTreeSet::new();
    let mut byte_count = name.len();

    for (record_index, record) in records.iter().enumerate() {
        if record.id.trim().is_empty() {
            return Err(VectorIndexError::InvalidRecordId {
                partition: name.clone(),
                record_index,
            });
        }
        if !seen.insert(record.id.clone()) {
            return Err(VectorIndexError::DuplicateRecordId {
                partition: name.clone(),
                id: record.id.clone(),
            });
        }
        if record.labels.keys().any(|key| key.trim().is_empty()) {
            return Err(VectorIndexError::InvalidLabel {
                context: format!("record '{}' in partition '{name}'", record.id),
            });
        }
        let context = format!("record '{}' in partition '{name}'", record.id);
        validate_vector(&record.embedding, descriptor, context)?;
        byte_count = accounted_record_bytes(byte_count, &record.id, &record.labels, descriptor)?;
        if byte_count > descriptor.max_bytes {
            return Err(VectorIndexError::BudgetExceeded {
                resource: VectorBudgetResource::Bytes,
                limit: descriptor.max_bytes,
                required: byte_count,
            });
        }
    }

    let vector_capacity = records
        .len()
        .checked_mul(descriptor.dimension)
        .ok_or(VectorIndexError::SizeOverflow)?;
    let mut ids = Vec::with_capacity(records.len());
    let mut labels = Vec::with_capacity(records.len());
    let mut vectors = Vec::with_capacity(vector_capacity);
    for record in records {
        let context = format!("record '{}' in partition '{name}'", record.id);
        let embedding = prepare_vector(record.embedding, descriptor, context)?;
        ids.push(record.id);
        labels.push(record.labels);
        vectors.extend(embedding);
    }

    Ok(Arc::new(PartitionBlock {
        name,
        ids,
        labels,
        vectors,
        byte_count,
    }))
}

fn accounted_record_bytes(
    current: usize,
    id: &str,
    labels: &BTreeMap<String, String>,
    descriptor: &VectorIndexDescriptor,
) -> VectorResult<usize> {
    let label_bytes = labels.iter().try_fold(0usize, |total, (key, value)| {
        total
            .checked_add(key.len())
            .and_then(|total| total.checked_add(value.len()))
            .ok_or(VectorIndexError::SizeOverflow)
    })?;
    let vector_bytes = descriptor
        .dimension
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(VectorIndexError::SizeOverflow)?;
    current
        .checked_add(id.len())
        .and_then(|value| value.checked_add(label_bytes))
        .and_then(|value| value.checked_add(vector_bytes))
        .ok_or(VectorIndexError::SizeOverflow)
}

pub(super) fn prepare_vector(
    mut vector: Vec<f32>,
    descriptor: &VectorIndexDescriptor,
    context: String,
) -> VectorResult<Vec<f32>> {
    validate_vector(&vector, descriptor, context.clone())?;
    if descriptor.normalization == VectorNormalization::Unit {
        normalize_unit(&mut vector);
    }
    Ok(vector)
}

fn validate_vector(
    vector: &[f32],
    descriptor: &VectorIndexDescriptor,
    context: String,
) -> VectorResult<()> {
    if vector.len() != descriptor.dimension {
        return Err(VectorIndexError::DimensionMismatch {
            context,
            expected: descriptor.dimension,
            actual: vector.len(),
        });
    }
    if let Some(element_index) = vector.iter().position(|value| !value.is_finite()) {
        return Err(VectorIndexError::NonFiniteVector {
            context,
            element_index,
        });
    }
    if descriptor.normalization == VectorNormalization::Unit {
        let squared_norm = vector.iter().fold(0.0f64, |sum, value| {
            let value = f64::from(*value);
            sum + value * value
        });
        if squared_norm == 0.0 {
            return Err(VectorIndexError::ZeroVector { context });
        }
    }
    Ok(())
}

fn normalize_unit(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .fold(0.0f64, |sum, value| {
            let value = f64::from(*value);
            sum + value * value
        })
        .sqrt();
    for value in vector {
        *value = (f64::from(*value) / norm) as f32;
    }
}

fn publish_partition(
    inner: &IndexInner,
    block: Arc<PartitionBlock>,
    expected_revision: Option<VectorRevision>,
) -> VectorResult<VectorIndexStatus> {
    let mut published = write_unpoisoned(&inner.snapshot);
    let current = Arc::clone(&published);
    verify_expected_revision(&current, expected_revision)?;
    let existing = current.partitions.get(&block.name);

    if block.record_count() == 0 && existing.is_none() {
        return Ok(current.status());
    }

    let old_records = existing.map_or(0, |partition| partition.record_count());
    let old_bytes = existing.map_or(0, |partition| partition.byte_count);
    let record_count = current
        .record_count
        .checked_sub(old_records)
        .and_then(|count| count.checked_add(block.record_count()))
        .ok_or(VectorIndexError::SizeOverflow)?;
    let retained_bytes = current
        .byte_count
        .checked_sub(old_bytes)
        .ok_or(VectorIndexError::SizeOverflow)?;
    let byte_count = if block.record_count() == 0 {
        retained_bytes
    } else {
        retained_bytes
            .checked_add(block.byte_count)
            .ok_or(VectorIndexError::SizeOverflow)?
    };
    enforce_budgets(&inner.descriptor, record_count, byte_count)?;

    let mut partitions = current.partitions.clone();
    if block.record_count() == 0 {
        partitions.remove(&block.name);
    } else {
        partitions.insert(block.name.clone(), block);
    }
    let next = Arc::new(IndexSnapshot {
        revision: current.revision.next()?,
        partitions,
        record_count,
        byte_count,
    });
    let status = next.status();
    *published = next;
    Ok(status)
}

fn remove_partition(
    inner: &IndexInner,
    partition: &str,
    expected_revision: Option<VectorRevision>,
) -> VectorResult<VectorIndexStatus> {
    let mut published = write_unpoisoned(&inner.snapshot);
    let current = Arc::clone(&published);
    verify_expected_revision(&current, expected_revision)?;
    let Some(existing) = current.partitions.get(partition) else {
        return Ok(current.status());
    };
    let mut partitions = current.partitions.clone();
    partitions.remove(partition);
    let next = Arc::new(IndexSnapshot {
        revision: current.revision.next()?,
        partitions,
        record_count: current
            .record_count
            .checked_sub(existing.record_count())
            .ok_or(VectorIndexError::SizeOverflow)?,
        byte_count: current
            .byte_count
            .checked_sub(existing.byte_count)
            .ok_or(VectorIndexError::SizeOverflow)?,
    });
    let status = next.status();
    *published = next;
    Ok(status)
}

fn verify_expected_revision(
    current: &IndexSnapshot,
    expected_revision: Option<VectorRevision>,
) -> VectorResult<()> {
    if let Some(expected) = expected_revision {
        if current.revision != expected {
            return Err(VectorIndexError::RevisionConflict {
                expected,
                actual: current.revision,
            });
        }
    }
    Ok(())
}

fn clear_index(inner: &IndexInner) -> VectorResult<VectorIndexStatus> {
    let mut published = write_unpoisoned(&inner.snapshot);
    let current = Arc::clone(&published);
    if current.partitions.is_empty() {
        return Ok(current.status());
    }
    let next = Arc::new(IndexSnapshot {
        revision: current.revision.next()?,
        ..IndexSnapshot::default()
    });
    let status = next.status();
    *published = next;
    Ok(status)
}

pub(super) fn enforce_budgets(
    descriptor: &VectorIndexDescriptor,
    record_count: usize,
    byte_count: usize,
) -> VectorResult<()> {
    if record_count > descriptor.max_records {
        return Err(VectorIndexError::BudgetExceeded {
            resource: VectorBudgetResource::Records,
            limit: descriptor.max_records,
            required: record_count,
        });
    }
    if byte_count > descriptor.max_bytes {
        return Err(VectorIndexError::BudgetExceeded {
            resource: VectorBudgetResource::Bytes,
            limit: descriptor.max_bytes,
            required: byte_count,
        });
    }
    Ok(())
}

fn read_unpoisoned<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_unpoisoned<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod lifetime_tests {
    use super::*;

    #[test]
    fn last_index_handle_releases_the_complete_index_graph() {
        let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(3)).unwrap();
        let clone = index.clone();
        let weak = Arc::downgrade(&index.inner);

        drop(index);
        assert!(weak.upgrade().is_some());
        drop(clone);
        assert!(weak.upgrade().is_none());
    }
}
