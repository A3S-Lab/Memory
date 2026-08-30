use super::super::in_memory::{enforce_budgets, new_history_digest, prepare_vector};
use super::super::search::search_snapshot;
use super::super::{
    VectorIndexChangeToken, VectorIndexDescriptor, VectorIndexError, VectorIndexObservation,
    VectorIndexStatus, VectorResult, VectorRevision, VectorSearchRequest, VectorSearchResult,
};
use super::snapshot::load_snapshot;
use rusqlite::{params, Connection, TransactionBehavior};
use std::path::Path;
use std::time::Duration;

const STORAGE_PROFILE: &str = "a3s.memory.sqlite-vector-index.v1";
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

const SCHEMA: &str = r#"
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;

CREATE TABLE IF NOT EXISTS a3s_vector_index_metadata (
    singleton       INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
    storage_profile TEXT    NOT NULL,
    descriptor_json TEXT    NOT NULL,
    history_digest  TEXT    NOT NULL,
    revision        TEXT    NOT NULL,
    partition_count INTEGER NOT NULL CHECK (partition_count >= 0),
    record_count    INTEGER NOT NULL CHECK (record_count >= 0),
    byte_count      INTEGER NOT NULL CHECK (byte_count >= 0)
);

CREATE TABLE IF NOT EXISTS a3s_vector_partitions (
    name           TEXT    PRIMARY KEY NOT NULL,
    record_count   INTEGER NOT NULL CHECK (record_count > 0),
    byte_count     INTEGER NOT NULL CHECK (byte_count > 0),
    content_digest TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS a3s_vector_records (
    partition  TEXT    NOT NULL,
    position   INTEGER NOT NULL CHECK (position >= 0),
    id         TEXT    NOT NULL,
    labels_json TEXT   NOT NULL,
    embedding  BLOB    NOT NULL,
    PRIMARY KEY (partition, position),
    UNIQUE (partition, id),
    FOREIGN KEY (partition) REFERENCES a3s_vector_partitions(name) ON DELETE CASCADE
);
"#;

pub(super) fn open(
    path: &Path,
    descriptor: &VectorIndexDescriptor,
) -> VectorResult<(Connection, VectorIndexObservation)> {
    let mut connection = Connection::open(path).map_err(|_| {
        VectorIndexError::StorageFailed("could not open the SQLite database".to_string())
    })?;
    connection.busy_timeout(BUSY_TIMEOUT).map_err(|_| {
        VectorIndexError::StorageFailed("could not configure SQLite lock waiting".to_string())
    })?;
    connection.execute_batch(SCHEMA).map_err(|_| {
        VectorIndexError::StorageFailed("could not initialize the SQLite schema".to_string())
    })?;

    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| {
            VectorIndexError::StorageFailed(
                "could not start SQLite index initialization".to_string(),
            )
        })?;
    initialize_metadata(&transaction, descriptor)?;
    let (_, observation) = load_snapshot(&transaction, descriptor)?;
    transaction.commit().map_err(|_| {
        VectorIndexError::StorageFailed("could not finish SQLite index initialization".to_string())
    })?;
    Ok((connection, observation))
}

pub(super) fn observe(
    connection: &mut Connection,
    descriptor: &VectorIndexDescriptor,
) -> VectorResult<VectorIndexObservation> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not start SQLite index observation".to_string())
        })?;
    let observation = read_observation(&transaction, descriptor)?;
    transaction.commit().map_err(|_| {
        VectorIndexError::StorageFailed("could not finish SQLite index observation".to_string())
    })?;
    Ok(observation)
}

pub(super) fn search(
    connection: &mut Connection,
    descriptor: &VectorIndexDescriptor,
    mut request: VectorSearchRequest,
) -> VectorResult<(VectorSearchResult, VectorIndexObservation)> {
    let query = prepare_vector(
        std::mem::take(&mut request.embedding),
        descriptor,
        "query".to_string(),
    )?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not start SQLite vector search".to_string())
        })?;
    let (snapshot, observation) = load_snapshot(&transaction, descriptor)?;
    transaction.commit().map_err(|_| {
        VectorIndexError::StorageFailed("could not finish SQLite vector search".to_string())
    })?;
    let result = search_snapshot(snapshot, descriptor, query, request)?;
    Ok((result, observation))
}

fn initialize_metadata(
    connection: &Connection,
    descriptor: &VectorIndexDescriptor,
) -> VectorResult<()> {
    let metadata_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM a3s_vector_index_metadata",
            [],
            |row| row.get(0),
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not inspect SQLite index metadata".to_string())
        })?;
    if metadata_count == 1 {
        return Ok(());
    }
    if metadata_count != 0 {
        return Err(corrupted("metadata singleton count is invalid"));
    }
    let content_rows = table_count(connection, "a3s_vector_partitions")?
        .checked_add(table_count(connection, "a3s_vector_records")?)
        .ok_or(VectorIndexError::SizeOverflow)?;
    if content_rows != 0 {
        return Err(corrupted("content exists without index metadata"));
    }

    let descriptor_json = serde_json::to_string(descriptor).map_err(|_| {
        VectorIndexError::StorageFailed("could not encode the vector descriptor".to_string())
    })?;
    connection
        .execute(
            "INSERT INTO a3s_vector_index_metadata
             (singleton, storage_profile, descriptor_json, history_digest, revision,
              partition_count, record_count, byte_count)
             VALUES (1, ?1, ?2, ?3, '0', 0, 0, 0)",
            params![STORAGE_PROFILE, descriptor_json, new_history_digest()],
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not create SQLite index metadata".to_string())
        })?;
    Ok(())
}

pub(super) fn read_observation(
    connection: &Connection,
    descriptor: &VectorIndexDescriptor,
) -> VectorResult<VectorIndexObservation> {
    let raw = connection
        .query_row(
            "SELECT storage_profile, descriptor_json, history_digest, revision,
                    partition_count, record_count, byte_count
             FROM a3s_vector_index_metadata WHERE singleton = 1",
            [],
            |row| {
                Ok(RawMetadata {
                    storage_profile: row.get(0)?,
                    descriptor_json: row.get(1)?,
                    history_digest: row.get(2)?,
                    revision: row.get(3)?,
                    partition_count: row.get(4)?,
                    record_count: row.get(5)?,
                    byte_count: row.get(6)?,
                })
            },
        )
        .map_err(|error| match error {
            rusqlite::Error::QueryReturnedNoRows => corrupted("index metadata is missing"),
            _ => {
                VectorIndexError::StorageFailed("could not read SQLite index metadata".to_string())
            }
        })?;
    if raw.storage_profile != STORAGE_PROFILE {
        return Err(corrupted("storage profile is unsupported"));
    }
    let stored_descriptor: VectorIndexDescriptor = serde_json::from_str(&raw.descriptor_json)
        .map_err(|_| corrupted("stored descriptor is not valid JSON"))?;
    stored_descriptor
        .validate()
        .map_err(|_| corrupted("stored descriptor violates index invariants"))?;
    if &stored_descriptor != descriptor {
        return Err(VectorIndexError::DescriptorMismatch);
    }
    let revision = raw
        .revision
        .parse::<u64>()
        .map(VectorRevision::new)
        .map_err(|_| corrupted("stored revision is invalid"))?;
    let status = VectorIndexStatus {
        revision,
        partition_count: nonnegative_usize(raw.partition_count, "partition count")?,
        record_count: nonnegative_usize(raw.record_count, "record count")?,
        byte_count: nonnegative_usize(raw.byte_count, "byte count")?,
    };
    enforce_budgets(descriptor, status.record_count, status.byte_count)
        .map_err(|_| corrupted("stored accounting exceeds the descriptor budgets"))?;
    let token = VectorIndexChangeToken::try_new(raw.history_digest, revision)
        .map_err(|_| corrupted("stored history identity is invalid"))?;
    let observation = VectorIndexObservation {
        status,
        change_token: Some(token),
    };
    observation
        .verify()
        .map_err(|_| corrupted("stored observation is inconsistent"))?;
    verify_aggregates(connection, &observation.status)?;
    Ok(observation)
}

fn verify_aggregates(connection: &Connection, status: &VectorIndexStatus) -> VectorResult<()> {
    let (partitions, records, bytes): (i64, i64, i64) = connection
        .query_row(
            "SELECT COUNT(*), COALESCE(SUM(record_count), 0), COALESCE(SUM(byte_count), 0)
             FROM a3s_vector_partitions",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not verify SQLite index accounting".to_string())
        })?;
    let actual_records = table_count(connection, "a3s_vector_records")?;
    let mismatched_partitions: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM a3s_vector_partitions AS p
             WHERE p.record_count <> (
                 SELECT COUNT(*) FROM a3s_vector_records AS r WHERE r.partition = p.name
             )",
            [],
            |row| row.get(0),
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed(
                "could not verify SQLite partition accounting".to_string(),
            )
        })?;
    let orphan_records: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM a3s_vector_records AS r
             LEFT JOIN a3s_vector_partitions AS p ON p.name = r.partition
             WHERE p.name IS NULL",
            [],
            |row| row.get(0),
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not verify SQLite record ownership".to_string())
        })?;

    if nonnegative_usize(partitions, "partition aggregate")? != status.partition_count
        || nonnegative_usize(records, "record aggregate")? != status.record_count
        || actual_records != status.record_count
        || nonnegative_usize(bytes, "byte aggregate")? != status.byte_count
        || mismatched_partitions != 0
        || orphan_records != 0
    {
        return Err(corrupted("stored accounting does not match index content"));
    }
    Ok(())
}

pub(super) fn table_count(connection: &Connection, table: &str) -> VectorResult<usize> {
    let sql = match table {
        "a3s_vector_partitions" => "SELECT COUNT(*) FROM a3s_vector_partitions",
        "a3s_vector_records" => "SELECT COUNT(*) FROM a3s_vector_records",
        _ => return Err(corrupted("an internal table selector is invalid")),
    };
    let count: i64 = connection
        .query_row(sql, [], |row| row.get(0))
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not count SQLite index rows".to_string())
        })?;
    nonnegative_usize(count, "table row count")
}

pub(super) fn nonnegative_usize(value: i64, name: &str) -> VectorResult<usize> {
    usize::try_from(value).map_err(|_| corrupted(&format!("{name} is outside the valid range")))
}

pub(super) fn sqlite_integer(value: usize) -> VectorResult<i64> {
    i64::try_from(value).map_err(|_| VectorIndexError::SizeOverflow)
}

pub(super) fn corrupted(message: &str) -> VectorIndexError {
    VectorIndexError::StorageCorrupted(message.to_string())
}

struct RawMetadata {
    storage_profile: String,
    descriptor_json: String,
    history_digest: String,
    revision: String,
    partition_count: i64,
    record_count: i64,
    byte_count: i64,
}
