use super::in_memory::{IndexSnapshot, PartitionBlock};
use super::{
    VectorIndexDescriptor, VectorIndexError, VectorResult, VectorSearchHit, VectorSearchRequest,
    VectorSearchResult,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;

pub(super) fn search_snapshot(
    snapshot: Arc<IndexSnapshot>,
    descriptor: &VectorIndexDescriptor,
    query: Vec<f32>,
    request: VectorSearchRequest,
) -> VectorResult<VectorSearchResult> {
    let mut candidates = BinaryHeap::new();
    let mut searched_records = 0usize;

    for (partition_name, partition) in &snapshot.partitions {
        if !request.partitions.is_empty() && !request.partitions.contains(partition_name) {
            continue;
        }
        for record_index in 0..partition.record_count() {
            if !labels_match(&partition.labels[record_index], &request.labels) {
                continue;
            }
            searched_records = searched_records
                .checked_add(1)
                .ok_or(VectorIndexError::SizeOverflow)?;
            let score =
                dot_product_f64(&query, partition.vector(record_index, descriptor.dimension));
            if !score.is_finite() || score > f64::from(f32::MAX) || score < f64::from(f32::MIN) {
                return Err(VectorIndexError::ScoreOverflow {
                    partition: partition_name.clone(),
                    id: partition.ids[record_index].clone(),
                });
            }
            candidates.push(ScoredRecord {
                partition,
                record_index,
                score,
            });
            if candidates.len() > request.limit {
                candidates.pop();
            }
        }
    }

    let mut ranked = candidates.into_vec();
    ranked.sort_by(compare_best_first);
    let hits = ranked
        .into_iter()
        .map(|candidate| VectorSearchHit {
            id: candidate.partition.ids[candidate.record_index].clone(),
            partition: candidate.partition.name.clone(),
            score: candidate.score as f32,
            labels: candidate.partition.labels[candidate.record_index].clone(),
        })
        .collect::<Vec<_>>();

    Ok(VectorSearchResult {
        truncated: searched_records > hits.len(),
        hits,
        status: snapshot.status(),
        searched_records,
    })
}

fn labels_match(
    record_labels: &BTreeMap<String, String>,
    required: &BTreeMap<String, String>,
) -> bool {
    required
        .iter()
        .all(|(key, value)| record_labels.get(key) == Some(value))
}

fn dot_product_f64(left: &[f32], right: &[f32]) -> f64 {
    left.iter().zip(right).fold(0.0f64, |score, (left, right)| {
        score + f64::from(*left) * f64::from(*right)
    })
}

struct ScoredRecord<'a> {
    partition: &'a PartitionBlock,
    record_index: usize,
    score: f64,
}

impl ScoredRecord<'_> {
    fn id(&self) -> &str {
        &self.partition.ids[self.record_index]
    }
}

impl PartialEq for ScoredRecord<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
            && self.partition.name == other.partition.name
            && self.id() == other.id()
    }
}

impl Eq for ScoredRecord<'_> {}

impl PartialOrd for ScoredRecord<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredRecord<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap keeps its greatest value at the root. Reverse score order
        // so the worst retained candidate is removed when the heap exceeds k.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.partition.name.cmp(&other.partition.name))
            .then_with(|| self.id().cmp(other.id()))
    }
}

fn compare_best_first(left: &ScoredRecord<'_>, right: &ScoredRecord<'_>) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.partition.name.cmp(&right.partition.name))
        .then_with(|| left.id().cmp(right.id()))
}
