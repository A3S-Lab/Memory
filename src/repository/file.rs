use super::{
    InMemoryRepository, MemoryAccessEvent, MemoryChangeResult, MemoryChangeSet, MemoryNamespace,
    MemoryNode, MemoryQuery, MemoryQueryResult, MemoryRepository, MemoryRepositoryError,
    MemoryUsageSummary,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

const JOURNAL_FILE: &str = "memory-v2.journal";
const LOCK_FILE: &str = "memory-v2.lock";
const JOURNAL_VERSION: u8 = 1;
const MAX_JOURNAL_RECORD_BYTES: usize = 16 * 1024 * 1024;

/// Durable local repository backed by a checksummed write-ahead journal.
///
/// One live instance owns a repository directory at a time. Mutations are
/// validated before they are appended and synced; only then are they published
/// to the in-memory read view. A restart deterministically replays the journal.
#[derive(Debug, Clone)]
pub struct FileMemoryRepository {
    state: Arc<FileRepositoryState>,
}

#[derive(Debug)]
struct FileRepositoryState {
    inner: InMemoryRepository,
    writer: Mutex<JournalWriter>,
    _lock_file: std::fs::File,
}

#[derive(Debug)]
struct JournalWriter {
    file: tokio::fs::File,
    poisoned: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "recordType", rename_all = "snake_case")]
enum JournalRecord {
    Change { change_set: MemoryChangeSet },
    Admission { event: MemoryAccessEvent },
    Use { event: MemoryAccessEvent },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JournalEnvelope {
    version: u8,
    checksum: String,
    record: JournalRecord,
}

impl FileMemoryRepository {
    pub async fn open(root: impl AsRef<Path>) -> Result<Self, MemoryRepositoryError> {
        let root = root.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&root)
            .await
            .map_err(|error| persistence("create repository directory", error))?;

        let lock_file = acquire_lock(root.join(LOCK_FILE)).await?;
        let journal_path = root.join(JOURNAL_FILE);
        let (records, valid_bytes, file_bytes) = read_journal(&journal_path).await?;
        if valid_bytes < file_bytes {
            truncate_torn_tail(&journal_path, valid_bytes).await?;
        }

        let inner = InMemoryRepository::new();
        for record in records {
            replay(&inner, record).await?;
        }

        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&journal_path)
            .await
            .map_err(|error| persistence("open journal for append", error))?;
        Ok(Self {
            state: Arc::new(FileRepositoryState {
                inner,
                writer: Mutex::new(JournalWriter {
                    file,
                    poisoned: false,
                }),
                _lock_file: lock_file,
            }),
        })
    }
}

impl FileRepositoryState {
    async fn persist_access(
        &self,
        event: MemoryAccessEvent,
        access_kind: AccessKind,
    ) -> Result<(), MemoryRepositoryError> {
        let mut writer = self.writer.lock().await;
        writer.ensure_healthy()?;
        let replayed = match access_kind {
            AccessKind::Admission => self.inner.preview_admission(&event).await?,
            AccessKind::Use => self.inner.preview_use(&event).await?,
        };
        if replayed {
            return Ok(());
        }

        let record = match access_kind {
            AccessKind::Admission => JournalRecord::Admission {
                event: event.clone(),
            },
            AccessKind::Use => JournalRecord::Use {
                event: event.clone(),
            },
        };
        writer.append(record).await?;
        let published = match access_kind {
            AccessKind::Admission => self.inner.record_admission(event).await,
            AccessKind::Use => self.inner.record_use(event).await,
        };
        if let Err(error) = published {
            writer.poisoned = true;
            return Err(MemoryRepositoryError::Persistence {
                operation: "publish synced access record".into(),
                message: error.to_string(),
            });
        }
        Ok(())
    }

    async fn apply_change(
        &self,
        change_set: MemoryChangeSet,
    ) -> Result<MemoryChangeResult, MemoryRepositoryError> {
        let mut writer = self.writer.lock().await;
        writer.ensure_healthy()?;
        let (expected, replayed) = self.inner.preview_apply(&change_set).await?;
        if replayed {
            return Ok(expected);
        }

        writer
            .append(JournalRecord::Change {
                change_set: change_set.clone(),
            })
            .await?;
        match self.inner.apply(change_set).await {
            Ok(actual) if actual == expected => Ok(actual),
            Ok(actual) => {
                writer.poisoned = true;
                Err(MemoryRepositoryError::Persistence {
                    operation: "publish synced change set".into(),
                    message: format!(
                        "preview result diverged from published result: expected {expected:?}, actual {actual:?}"
                    ),
                })
            }
            Err(error) => {
                writer.poisoned = true;
                Err(MemoryRepositoryError::Persistence {
                    operation: "publish synced change set".into(),
                    message: error.to_string(),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AccessKind {
    Admission,
    Use,
}

#[async_trait::async_trait]
impl MemoryRepository for FileMemoryRepository {
    async fn apply(
        &self,
        change_set: MemoryChangeSet,
    ) -> Result<MemoryChangeResult, MemoryRepositoryError> {
        let state = self.state.clone();
        join_transaction(
            tokio::spawn(async move { state.apply_change(change_set).await }),
            "apply change set",
        )
        .await
    }

    async fn get(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<Option<MemoryNode>, MemoryRepositoryError> {
        self.state.inner.get(namespace, node_id).await
    }

    async fn query(&self, query: MemoryQuery) -> Result<MemoryQueryResult, MemoryRepositoryError> {
        self.state.inner.query(query).await
    }

    async fn record_admission(
        &self,
        event: MemoryAccessEvent,
    ) -> Result<(), MemoryRepositoryError> {
        let state = self.state.clone();
        join_transaction(
            tokio::spawn(async move { state.persist_access(event, AccessKind::Admission).await }),
            "record admission",
        )
        .await
    }

    async fn record_use(&self, event: MemoryAccessEvent) -> Result<(), MemoryRepositoryError> {
        let state = self.state.clone();
        join_transaction(
            tokio::spawn(async move { state.persist_access(event, AccessKind::Use).await }),
            "record use",
        )
        .await
    }

    async fn usage_summary(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<MemoryUsageSummary, MemoryRepositoryError> {
        self.state.inner.usage_summary(namespace, node_id).await
    }
}

async fn join_transaction<T: Send + 'static>(
    handle: tokio::task::JoinHandle<Result<T, MemoryRepositoryError>>,
    operation: &str,
) -> Result<T, MemoryRepositoryError> {
    handle
        .await
        .map_err(|error| MemoryRepositoryError::Persistence {
            operation: operation.into(),
            message: format!("transaction task failed: {error}"),
        })?
}

impl JournalWriter {
    fn ensure_healthy(&self) -> Result<(), MemoryRepositoryError> {
        if self.poisoned {
            return Err(MemoryRepositoryError::Persistence {
                operation: "append journal".into(),
                message: "writer is poisoned; drop and reopen the repository".into(),
            });
        }
        Ok(())
    }

    async fn append(&mut self, record: JournalRecord) -> Result<(), MemoryRepositoryError> {
        self.ensure_healthy()?;
        let encoded = encode_record(record)?;
        if encoded.len() > MAX_JOURNAL_RECORD_BYTES {
            return Err(MemoryRepositoryError::LimitExceeded {
                resource: "journal record bytes".into(),
                limit: MAX_JOURNAL_RECORD_BYTES,
                actual: encoded.len(),
            });
        }
        if let Err(error) = self.file.write_all(&encoded).await {
            self.poisoned = true;
            return Err(persistence("write journal record", error));
        }
        if let Err(error) = self.file.write_all(b"\n").await {
            self.poisoned = true;
            return Err(persistence("write journal delimiter", error));
        }
        if let Err(error) = self.file.flush().await {
            self.poisoned = true;
            return Err(persistence("flush journal record", error));
        }
        if let Err(error) = self.file.sync_data().await {
            self.poisoned = true;
            return Err(persistence("sync journal record", error));
        }
        Ok(())
    }
}

async fn acquire_lock(path: PathBuf) -> Result<std::fs::File, MemoryRepositoryError> {
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        fs2::FileExt::try_lock_exclusive(&file)?;
        Ok::<_, std::io::Error>(file)
    })
    .await
    .map_err(|error| MemoryRepositoryError::Persistence {
        operation: "join repository lock task".into(),
        message: error.to_string(),
    })?
    .map_err(|error| persistence("acquire exclusive repository lock", error))
}

async fn read_journal(
    path: &Path,
) -> Result<(Vec<JournalRecord>, u64, u64), MemoryRepositoryError> {
    let file = match tokio::fs::File::open(path).await {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Vec::new(), 0, 0));
        }
        Err(error) => return Err(persistence("open journal for recovery", error)),
    };
    let file_bytes = file
        .metadata()
        .await
        .map_err(|error| persistence("read journal metadata", error))?
        .len();
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut valid_bytes = 0_u64;
    let mut line = Vec::new();

    loop {
        line.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .await
            .map_err(|error| persistence("read journal record", error))?;
        if bytes_read == 0 {
            break;
        }
        if line.len() > MAX_JOURNAL_RECORD_BYTES + 1 {
            return Err(MemoryRepositoryError::LimitExceeded {
                resource: "journal record bytes".into(),
                limit: MAX_JOURNAL_RECORD_BYTES,
                actual: line.len(),
            });
        }
        if line.last() != Some(&b'\n') {
            break;
        }
        line.pop();
        if line.is_empty() {
            return Err(MemoryRepositoryError::Persistence {
                operation: "decode journal record".into(),
                message: "empty journal record".into(),
            });
        }
        records.push(decode_record(&line)?);
        valid_bytes += bytes_read as u64;
    }
    Ok((records, valid_bytes, file_bytes))
}

async fn truncate_torn_tail(path: &Path, valid_bytes: u64) -> Result<(), MemoryRepositoryError> {
    let file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .await
        .map_err(|error| persistence("open torn journal for repair", error))?;
    file.set_len(valid_bytes)
        .await
        .map_err(|error| persistence("truncate torn journal tail", error))?;
    file.sync_data()
        .await
        .map_err(|error| persistence("sync repaired journal", error))
}

async fn replay(
    repository: &InMemoryRepository,
    record: JournalRecord,
) -> Result<(), MemoryRepositoryError> {
    let result = match record {
        JournalRecord::Change { change_set } => repository.apply(change_set).await.map(|_| ()),
        JournalRecord::Admission { event } => repository.record_admission(event).await,
        JournalRecord::Use { event } => repository.record_use(event).await,
    };
    result.map_err(|error| MemoryRepositoryError::Persistence {
        operation: "replay journal record".into(),
        message: error.to_string(),
    })
}

fn encode_record(record: JournalRecord) -> Result<Vec<u8>, MemoryRepositoryError> {
    let checksum = record_checksum(&record)?;
    serde_json::to_vec(&JournalEnvelope {
        version: JOURNAL_VERSION,
        checksum,
        record,
    })
    .map_err(|error| MemoryRepositoryError::Persistence {
        operation: "encode journal record".into(),
        message: error.to_string(),
    })
}

fn decode_record(bytes: &[u8]) -> Result<JournalRecord, MemoryRepositoryError> {
    let envelope = serde_json::from_slice::<JournalEnvelope>(bytes).map_err(|error| {
        MemoryRepositoryError::Persistence {
            operation: "decode journal record".into(),
            message: error.to_string(),
        }
    })?;
    if envelope.version != JOURNAL_VERSION {
        return Err(MemoryRepositoryError::Persistence {
            operation: "decode journal record".into(),
            message: format!("unsupported journal version: {}", envelope.version),
        });
    }
    let actual = record_checksum(&envelope.record)?;
    if actual != envelope.checksum {
        return Err(MemoryRepositoryError::Persistence {
            operation: "verify journal checksum".into(),
            message: format!(
                "checksum mismatch: expected {}, actual {actual}",
                envelope.checksum
            ),
        });
    }
    Ok(envelope.record)
}

fn record_checksum(record: &JournalRecord) -> Result<String, MemoryRepositoryError> {
    let bytes = serde_json::to_vec(record).map_err(|error| MemoryRepositoryError::Persistence {
        operation: "encode journal checksum payload".into(),
        message: error.to_string(),
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn persistence(operation: &str, error: impl std::fmt::Display) -> MemoryRepositoryError {
    MemoryRepositoryError::Persistence {
        operation: operation.into(),
        message: error.to_string(),
    }
}
