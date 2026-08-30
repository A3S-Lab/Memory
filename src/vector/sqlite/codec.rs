use super::super::in_memory::PartitionBlock;
use super::super::{VectorIndexError, VectorRecord, VectorResult};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const PARTITION_DIGEST_DOMAIN: &str = "a3s.memory.sqlite-vector-partition.v1";

pub(super) fn encode_vector(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(super) fn decode_vector(blob: &[u8], dimension: usize) -> VectorResult<Vec<f32>> {
    let expected_bytes = dimension
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(VectorIndexError::SizeOverflow)?;
    if blob.len() != expected_bytes {
        return Err(VectorIndexError::StorageCorrupted(
            "stored vector has an invalid byte length".to_string(),
        ));
    }
    let mut vector = Vec::with_capacity(dimension);
    for bytes in blob.chunks_exact(std::mem::size_of::<f32>()) {
        vector.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    }
    Ok(vector)
}

pub(super) fn digest_block(block: &PartitionBlock, dimension: usize) -> String {
    let mut hasher = partition_hasher(&block.name, block.record_count());
    for ((id, labels), vector) in block
        .ids
        .iter()
        .zip(&block.labels)
        .zip(block.vectors.chunks_exact(dimension))
    {
        digest_record(&mut hasher, id, labels, vector);
    }
    format!("sha256:{:x}", hasher.finalize())
}

pub(super) fn digest_records(partition: &str, records: &[VectorRecord]) -> String {
    let mut hasher = partition_hasher(partition, records.len());
    for record in records {
        digest_record(&mut hasher, &record.id, &record.labels, &record.embedding);
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn partition_hasher(partition: &str, record_count: usize) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(PARTITION_DIGEST_DOMAIN.as_bytes());
    digest_bytes(&mut hasher, partition.as_bytes());
    hasher.update((record_count as u64).to_le_bytes());
    hasher
}

fn digest_record(hasher: &mut Sha256, id: &str, labels: &BTreeMap<String, String>, vector: &[f32]) {
    digest_bytes(hasher, id.as_bytes());
    hasher.update((labels.len() as u64).to_le_bytes());
    for (key, value) in labels {
        digest_bytes(hasher, key.as_bytes());
        digest_bytes(hasher, value.as_bytes());
    }
    hasher.update((vector.len() as u64).to_le_bytes());
    for value in vector {
        hasher.update(value.to_le_bytes());
    }
}

fn digest_bytes(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}
