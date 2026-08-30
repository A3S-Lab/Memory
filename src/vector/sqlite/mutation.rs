use super::super::in_memory::{build_partition, enforce_budgets, PartitionBlock};
use super::super::{
    VectorIndexDescriptor, VectorIndexError, VectorIndexObservation, VectorIndexStatus,
    VectorRecord, VectorResult, VectorRevision,
};
use super::codec::{digest_block, encode_vector};
use super::storage::{corrupted, nonnegative_usize, read_observation, sqlite_integer};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use std::sync::Arc;

pub(super) struct MutationOutcome {
    pub(super) observation: VectorIndexObservation,
    pub(super) result: VectorResult<VectorIndexStatus>,
}

pub(super) fn replace_partition(
    connection: &mut Connection,
    descriptor: &VectorIndexDescriptor,
    partition: String,
    records: Vec<VectorRecord>,
    expected_revision: Option<VectorRevision>,
) -> VectorResult<MutationOutcome> {
    let block = build_partition(descriptor, partition, records)?;
    let transaction = begin_mutation(connection)?;
    let current = read_observation(&transaction, descriptor)?;
    if let Some(outcome) = revision_conflict(&current, expected_revision) {
        finish_read_only(transaction)?;
        return Ok(outcome);
    }

    let existing = partition_accounting(&transaction, &block.name)?;
    if block.record_count() == 0 && existing.is_none() {
        finish_read_only(transaction)?;
        return Ok(success(current));
    }

    let (old_records, old_bytes) = existing.unwrap_or_default();
    let record_count = current
        .status
        .record_count
        .checked_sub(old_records)
        .and_then(|count| count.checked_add(block.record_count()))
        .ok_or(VectorIndexError::SizeOverflow)?;
    let retained_bytes = current
        .status
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
    enforce_budgets(descriptor, record_count, byte_count)?;

    delete_partition(&transaction, &block.name)?;
    if block.record_count() > 0 {
        insert_partition(&transaction, descriptor, &block)?;
    }
    let partition_count = match (existing.is_some(), block.record_count() > 0) {
        (false, true) => current
            .status
            .partition_count
            .checked_add(1)
            .ok_or(VectorIndexError::SizeOverflow)?,
        (true, false) => current
            .status
            .partition_count
            .checked_sub(1)
            .ok_or(VectorIndexError::SizeOverflow)?,
        _ => current.status.partition_count,
    };
    let next = advance_observation(
        &current,
        VectorIndexStatus {
            revision: current.status.revision.next()?,
            partition_count,
            record_count,
            byte_count,
        },
    )?;
    update_status(&transaction, &next.status)?;
    commit_mutation(transaction)?;
    Ok(success(next))
}

pub(super) fn remove_partition(
    connection: &mut Connection,
    descriptor: &VectorIndexDescriptor,
    partition: String,
    expected_revision: Option<VectorRevision>,
) -> VectorResult<MutationOutcome> {
    let transaction = begin_mutation(connection)?;
    let current = read_observation(&transaction, descriptor)?;
    if let Some(outcome) = revision_conflict(&current, expected_revision) {
        finish_read_only(transaction)?;
        return Ok(outcome);
    }
    let Some((old_records, old_bytes)) = partition_accounting(&transaction, &partition)? else {
        finish_read_only(transaction)?;
        return Ok(success(current));
    };

    delete_partition(&transaction, &partition)?;
    let next = advance_observation(
        &current,
        VectorIndexStatus {
            revision: current.status.revision.next()?,
            partition_count: current
                .status
                .partition_count
                .checked_sub(1)
                .ok_or(VectorIndexError::SizeOverflow)?,
            record_count: current
                .status
                .record_count
                .checked_sub(old_records)
                .ok_or(VectorIndexError::SizeOverflow)?,
            byte_count: current
                .status
                .byte_count
                .checked_sub(old_bytes)
                .ok_or(VectorIndexError::SizeOverflow)?,
        },
    )?;
    update_status(&transaction, &next.status)?;
    commit_mutation(transaction)?;
    Ok(success(next))
}

pub(super) fn clear(
    connection: &mut Connection,
    descriptor: &VectorIndexDescriptor,
) -> VectorResult<MutationOutcome> {
    let transaction = begin_mutation(connection)?;
    let current = read_observation(&transaction, descriptor)?;
    if current.status.partition_count == 0 {
        finish_read_only(transaction)?;
        return Ok(success(current));
    }
    transaction
        .execute("DELETE FROM a3s_vector_partitions", [])
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not clear SQLite vector partitions".to_string())
        })?;
    let next = advance_observation(
        &current,
        VectorIndexStatus {
            revision: current.status.revision.next()?,
            ..VectorIndexStatus::default()
        },
    )?;
    update_status(&transaction, &next.status)?;
    commit_mutation(transaction)?;
    Ok(success(next))
}

fn insert_partition(
    connection: &Connection,
    descriptor: &VectorIndexDescriptor,
    block: &Arc<PartitionBlock>,
) -> VectorResult<()> {
    connection
        .execute(
            "INSERT INTO a3s_vector_partitions
             (name, record_count, byte_count, content_digest) VALUES (?1, ?2, ?3, ?4)",
            params![
                &block.name,
                sqlite_integer(block.record_count())?,
                sqlite_integer(block.byte_count)?,
                digest_block(block, descriptor.dimension),
            ],
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed(
                "could not publish a SQLite vector partition".to_string(),
            )
        })?;

    for (position, ((id, labels), vector)) in block
        .ids
        .iter()
        .zip(&block.labels)
        .zip(block.vectors.chunks_exact(descriptor.dimension))
        .enumerate()
    {
        let labels_json = serde_json::to_string(labels).map_err(|_| {
            VectorIndexError::StorageFailed("could not encode vector labels".to_string())
        })?;
        connection
            .execute(
                "INSERT INTO a3s_vector_records
                 (partition, position, id, labels_json, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    &block.name,
                    sqlite_integer(position)?,
                    id,
                    labels_json,
                    encode_vector(vector),
                ],
            )
            .map_err(|_| {
                VectorIndexError::StorageFailed("could not publish SQLite vector data".to_string())
            })?;
    }
    Ok(())
}

fn begin_mutation(connection: &mut Connection) -> VectorResult<rusqlite::Transaction<'_>> {
    connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not start SQLite vector mutation".to_string())
        })
}

fn commit_mutation(transaction: rusqlite::Transaction<'_>) -> VectorResult<()> {
    transaction.commit().map_err(|_| {
        VectorIndexError::StorageFailed("could not commit SQLite vector mutation".to_string())
    })
}

fn finish_read_only(transaction: rusqlite::Transaction<'_>) -> VectorResult<()> {
    transaction.commit().map_err(|_| {
        VectorIndexError::StorageFailed("could not finish SQLite vector check".to_string())
    })
}

fn partition_accounting(
    connection: &Connection,
    partition: &str,
) -> VectorResult<Option<(usize, usize)>> {
    let accounting = connection
        .query_row(
            "SELECT record_count, byte_count FROM a3s_vector_partitions WHERE name = ?1",
            params![partition],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(|_| {
            VectorIndexError::StorageFailed(
                "could not inspect SQLite partition accounting".to_string(),
            )
        })?;
    accounting
        .map(|(records, bytes)| {
            Ok((
                nonnegative_usize(records, "partition record count")?,
                nonnegative_usize(bytes, "partition byte count")?,
            ))
        })
        .transpose()
}

fn delete_partition(connection: &Connection, partition: &str) -> VectorResult<()> {
    connection
        .execute(
            "DELETE FROM a3s_vector_partitions WHERE name = ?1",
            params![partition],
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed(
                "could not remove a SQLite vector partition".to_string(),
            )
        })?;
    Ok(())
}

fn update_status(connection: &Connection, status: &VectorIndexStatus) -> VectorResult<()> {
    let changed = connection
        .execute(
            "UPDATE a3s_vector_index_metadata
             SET revision = ?1, partition_count = ?2, record_count = ?3, byte_count = ?4
             WHERE singleton = 1",
            params![
                status.revision.value().to_string(),
                sqlite_integer(status.partition_count)?,
                sqlite_integer(status.record_count)?,
                sqlite_integer(status.byte_count)?,
            ],
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not update SQLite index metadata".to_string())
        })?;
    if changed != 1 {
        return Err(corrupted("metadata update did not affect one row"));
    }
    Ok(())
}

fn advance_observation(
    current: &VectorIndexObservation,
    status: VectorIndexStatus,
) -> VectorResult<VectorIndexObservation> {
    let token = current
        .change_token
        .as_ref()
        .ok_or_else(|| corrupted("durable history token is missing"))?
        .with_revision(status.revision);
    let observation = VectorIndexObservation {
        status,
        change_token: Some(token),
    };
    observation.verify()?;
    Ok(observation)
}

fn revision_conflict(
    current: &VectorIndexObservation,
    expected: Option<VectorRevision>,
) -> Option<MutationOutcome> {
    expected
        .filter(|expected| *expected != current.status.revision)
        .map(|expected| MutationOutcome {
            observation: current.clone(),
            result: Err(VectorIndexError::RevisionConflict {
                expected,
                actual: current.status.revision,
            }),
        })
}

fn success(observation: VectorIndexObservation) -> MutationOutcome {
    MutationOutcome {
        result: Ok(observation.status.clone()),
        observation,
    }
}
