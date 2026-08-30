use super::super::in_memory::{build_partition, IndexSnapshot};
use super::super::{
    VectorIndexDescriptor, VectorIndexError, VectorIndexObservation, VectorRecord, VectorResult,
};
use super::codec::{decode_vector, digest_records};
use super::storage::{corrupted, nonnegative_usize, read_observation};
use rusqlite::{params, Connection};
use std::collections::BTreeMap;
use std::sync::Arc;

pub(super) fn load_snapshot(
    connection: &Connection,
    descriptor: &VectorIndexDescriptor,
) -> VectorResult<(Arc<IndexSnapshot>, VectorIndexObservation)> {
    let observation = read_observation(connection, descriptor)?;
    let mut statement = connection
        .prepare(
            "SELECT name, record_count, byte_count, content_digest
             FROM a3s_vector_partitions ORDER BY name",
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not read SQLite vector partitions".to_string())
        })?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not scan SQLite vector partitions".to_string())
        })?;
    let mut stored_partitions = Vec::new();
    for row in rows {
        stored_partitions.push(row.map_err(|_| {
            VectorIndexError::StorageFailed("could not decode SQLite partition data".to_string())
        })?);
    }
    drop(statement);

    let mut partitions = BTreeMap::new();
    for (name, stored_records, stored_bytes, stored_digest) in stored_partitions {
        let records = load_records(connection, descriptor, &name)?;
        if digest_records(&name, &records) != stored_digest {
            return Err(corrupted(
                "partition content digest does not match stored rows",
            ));
        }
        let block = build_partition(descriptor, name.clone(), records)
            .map_err(|_| corrupted("stored partition violates vector invariants"))?;
        if block.record_count() != nonnegative_usize(stored_records, "partition record count")?
            || block.byte_count != nonnegative_usize(stored_bytes, "partition byte count")?
        {
            return Err(corrupted("partition accounting does not match stored rows"));
        }
        if partitions.insert(name, block).is_some() {
            return Err(corrupted("duplicate partition metadata was found"));
        }
    }

    let snapshot = Arc::new(IndexSnapshot {
        revision: observation.status.revision,
        partitions,
        record_count: observation.status.record_count,
        byte_count: observation.status.byte_count,
    });
    if snapshot.status() != observation.status {
        return Err(corrupted("loaded snapshot does not match index metadata"));
    }
    Ok((snapshot, observation))
}

fn load_records(
    connection: &Connection,
    descriptor: &VectorIndexDescriptor,
    partition: &str,
) -> VectorResult<Vec<VectorRecord>> {
    let mut statement = connection
        .prepare(
            "SELECT position, id, labels_json, embedding
             FROM a3s_vector_records WHERE partition = ?1 ORDER BY position",
        )
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not read SQLite vector records".to_string())
        })?;
    let rows = statement
        .query_map(params![partition], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Vec<u8>>(3)?,
            ))
        })
        .map_err(|_| {
            VectorIndexError::StorageFailed("could not scan SQLite vector records".to_string())
        })?;
    let mut records = Vec::new();
    for (expected_position, row) in rows.enumerate() {
        let (position, id, labels_json, embedding) = row.map_err(|_| {
            VectorIndexError::StorageFailed("could not decode SQLite vector data".to_string())
        })?;
        if nonnegative_usize(position, "record position")? != expected_position {
            return Err(corrupted("record positions are not contiguous"));
        }
        let labels: BTreeMap<String, String> = serde_json::from_str(&labels_json)
            .map_err(|_| corrupted("stored labels are not valid JSON"))?;
        records.push(VectorRecord {
            id,
            embedding: decode_vector(&embedding, descriptor.dimension)?,
            labels,
        });
    }
    Ok(records)
}
