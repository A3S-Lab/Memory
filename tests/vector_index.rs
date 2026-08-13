use a3s_memory::{
    InMemoryVectorIndex, VectorBudgetResource, VectorIndex, VectorIndexDescriptor,
    VectorIndexError, VectorMetric, VectorNormalization, VectorRecord, VectorSearchRequest,
};
use std::sync::Arc;

fn record(id: &str, embedding: &[f32]) -> VectorRecord {
    VectorRecord::new(id, embedding.to_vec())
}

#[test]
fn vector_index_is_object_safe_send_and_sync() {
    fn assert_send_sync<T: Send + Sync + ?Sized>() {}
    assert_send_sync::<dyn VectorIndex>();

    let index: Arc<dyn VectorIndex> =
        Arc::new(InMemoryVectorIndex::new(VectorIndexDescriptor::new(3)).expect("valid index"));
    assert_eq!(index.descriptor().dimension, 3);
}

#[test]
fn descriptor_rejects_one_vector_larger_than_the_byte_budget() {
    let error =
        InMemoryVectorIndex::new(VectorIndexDescriptor::new(3).with_max_bytes(11)).unwrap_err();
    assert!(matches!(error, VectorIndexError::InvalidDescriptor(_)));
}

#[tokio::test]
async fn cosine_search_is_exact_and_dimension_is_dynamic() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(3)).unwrap();
    let revision = index
        .replace_partition(
            "src/lib.rs",
            vec![
                record("exact", &[2.0, 0.0, 0.0]),
                record("near", &[1.0, 1.0, 0.0]),
                record("opposite", &[-1.0, 0.0, 0.0]),
            ],
        )
        .await
        .unwrap();

    assert_eq!(revision.revision.value(), 1);
    assert_eq!(revision.partition_count, 1);
    assert_eq!(revision.record_count, 3);

    let result = index
        .search(VectorSearchRequest::new(vec![5.0, 0.0, 0.0], 3))
        .await
        .unwrap();

    assert_eq!(
        result
            .hits
            .iter()
            .map(|hit| hit.id.as_str())
            .collect::<Vec<_>>(),
        ["exact", "near", "opposite"]
    );
    assert!((result.hits[0].score - 1.0).abs() < 1e-6);
    assert_eq!(result.status.revision.value(), 1);
    assert_eq!(result.searched_records, 3);
    assert!(!result.truncated);
}

#[tokio::test]
async fn partition_replacement_and_removal_are_atomic_revisions() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    index
        .replace_partition("a", vec![record("old", &[1.0, 0.0])])
        .await
        .unwrap();
    index
        .replace_partition("b", vec![record("stable", &[0.0, 1.0])])
        .await
        .unwrap();

    let replaced = index
        .replace_partition("a", vec![record("new", &[1.0, 1.0])])
        .await
        .unwrap();
    assert_eq!(replaced.revision.value(), 3);
    assert_eq!(replaced.record_count, 2);

    let result = index
        .search(VectorSearchRequest::new(vec![1.0, 0.0], 10))
        .await
        .unwrap();
    let ids = result
        .hits
        .iter()
        .map(|hit| hit.id.as_str())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"new"));
    assert!(ids.contains(&"stable"));
    assert!(!ids.contains(&"old"));

    let removed = index.remove_partition("a").await.unwrap();
    assert_eq!(removed.revision.value(), 4);
    assert_eq!(removed.partition_count, 1);
    assert_eq!(removed.record_count, 1);

    let unchanged = index.remove_partition("missing").await.unwrap();
    assert_eq!(unchanged.revision.value(), 4);
}

#[tokio::test]
async fn empty_replacement_removes_partition_and_all_accounted_bytes() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    let populated = index
        .replace_partition("source", vec![record("record", &[1.0, 0.0])])
        .await
        .unwrap();
    assert!(populated.byte_count > 0);

    let empty = index.replace_partition("source", Vec::new()).await.unwrap();
    assert_eq!(empty.revision.value(), 2);
    assert_eq!(empty.partition_count, 0);
    assert_eq!(empty.record_count, 0);
    assert_eq!(empty.byte_count, 0);
}

#[tokio::test]
async fn search_filters_partitions_and_exact_labels() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    index
        .replace_partition(
            "src",
            vec![
                record("rust", &[1.0, 0.0]).with_label("language", "rust"),
                record("docs", &[1.0, 0.0]).with_label("language", "markdown"),
            ],
        )
        .await
        .unwrap();
    index
        .replace_partition(
            "tests",
            vec![record("test", &[1.0, 0.0]).with_label("language", "rust")],
        )
        .await
        .unwrap();

    let result = index
        .search(
            VectorSearchRequest::new(vec![1.0, 0.0], 10)
                .with_partition("src")
                .with_label("language", "rust"),
        )
        .await
        .unwrap();

    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, "rust");
    assert_eq!(result.hits[0].partition, "src");
    assert_eq!(result.searched_records, 1);
}

#[tokio::test]
async fn equal_scores_have_deterministic_partition_and_record_order() {
    let descriptor = VectorIndexDescriptor::new(2)
        .with_metric(VectorMetric::DotProduct)
        .with_normalization(VectorNormalization::None);
    let index = InMemoryVectorIndex::new(descriptor).unwrap();
    index
        .replace_partition(
            "z",
            vec![record("b", &[1.0, 0.0]), record("a", &[1.0, 0.0])],
        )
        .await
        .unwrap();
    index
        .replace_partition("a", vec![record("c", &[1.0, 0.0])])
        .await
        .unwrap();

    let result = index
        .search(VectorSearchRequest::new(vec![1.0, 0.0], 10))
        .await
        .unwrap();
    let keys = result
        .hits
        .iter()
        .map(|hit| format!("{}/{}", hit.partition, hit.id))
        .collect::<Vec<_>>();
    assert_eq!(keys, ["a/c", "z/a", "z/b"]);
}

#[tokio::test]
async fn invalid_vectors_and_duplicate_ids_fail_closed() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();

    assert!(matches!(
        index
            .replace_partition("src", vec![record("short", &[1.0])])
            .await,
        Err(VectorIndexError::DimensionMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    assert!(matches!(
        index
            .replace_partition("src", vec![record("nan", &[f32::NAN, 0.0])])
            .await,
        Err(VectorIndexError::NonFiniteVector { .. })
    ));
    assert!(matches!(
        index
            .replace_partition("src", vec![record("zero", &[0.0, 0.0])])
            .await,
        Err(VectorIndexError::ZeroVector { .. })
    ));
    assert!(matches!(
        index
            .replace_partition(
                "src",
                vec![record("same", &[1.0, 0.0]), record("same", &[0.0, 1.0])],
            )
            .await,
        Err(VectorIndexError::DuplicateRecordId { .. })
    ));
    assert_eq!(index.status().revision.value(), 0);
    assert_eq!(index.status().record_count, 0);
}

#[tokio::test]
async fn record_and_byte_budgets_are_enforced_without_mutation() {
    let record_limited = InMemoryVectorIndex::new(
        VectorIndexDescriptor::new(2)
            .with_max_records(1)
            .with_max_bytes(1024 * 1024),
    )
    .unwrap();
    let error = record_limited
        .replace_partition(
            "src",
            vec![record("a", &[1.0, 0.0]), record("b", &[0.0, 1.0])],
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        VectorIndexError::BudgetExceeded {
            resource: VectorBudgetResource::Records,
            limit: 1,
            required: 2
        }
    ));
    assert_eq!(record_limited.status().record_count, 0);

    let byte_limited = InMemoryVectorIndex::new(
        VectorIndexDescriptor::new(2)
            .with_max_records(10)
            .with_max_bytes(8),
    )
    .unwrap();
    let error = byte_limited
        .replace_partition("src", vec![record("a", &[1.0, 0.0])])
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        VectorIndexError::BudgetExceeded {
            resource: VectorBudgetResource::Bytes,
            limit: 8,
            ..
        }
    ));
    assert_eq!(byte_limited.status().record_count, 0);
}

#[tokio::test]
async fn zero_limit_and_invalid_query_leave_index_unchanged() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    index
        .replace_partition("src", vec![record("a", &[1.0, 0.0])])
        .await
        .unwrap();

    assert!(matches!(
        index
            .search(VectorSearchRequest::new(vec![1.0, 0.0], 0))
            .await,
        Err(VectorIndexError::InvalidRequest(_))
    ));
    assert!(matches!(
        index.search(VectorSearchRequest::new(vec![1.0], 1)).await,
        Err(VectorIndexError::DimensionMismatch { .. })
    ));
    assert_eq!(index.status().revision.value(), 1);
}

#[tokio::test]
async fn exact_top_k_matches_independent_brute_force_oracle() {
    const DIMENSION: usize = 17;
    const RECORDS: usize = 257;
    const LIMIT: usize = 37;

    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(DIMENSION)).unwrap();
    let mut rng = DeterministicRng::new(0x5eed_1234_9876_abcd);
    let mut originals = Vec::new();
    let mut records = Vec::new();
    for record_index in 0..RECORDS {
        let embedding = (0..DIMENSION)
            .map(|_| rng.next_signed_f32())
            .collect::<Vec<_>>();
        let id = format!("record-{record_index:04}");
        originals.push((id.clone(), embedding.clone()));
        records.push(record(&id, &embedding));
    }
    index.replace_partition("oracle", records).await.unwrap();

    let query = (0..DIMENSION)
        .map(|_| rng.next_signed_f32())
        .collect::<Vec<_>>();
    let actual = index
        .search(VectorSearchRequest::new(query.clone(), LIMIT))
        .await
        .unwrap();
    let expected = brute_force_cosine(&query, &originals, LIMIT);

    assert_eq!(actual.hits.len(), LIMIT);
    assert_eq!(actual.searched_records, RECORDS);
    assert!(actual.truncated);
    for (hit, (expected_id, expected_score)) in actual.hits.iter().zip(expected) {
        assert_eq!(hit.id, expected_id);
        assert!(
            (f64::from(hit.score) - expected_score).abs() < 2e-6,
            "{}: actual={}, expected={expected_score}",
            hit.id,
            hit.score
        );
    }
}

#[tokio::test]
async fn concurrent_search_observes_only_complete_partition_replacements() {
    let index = Arc::new(InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap());
    index
        .replace_partition(
            "active",
            vec![
                record("generation-0:a", &[1.0, 0.0]),
                record("generation-0:b", &[0.0, 1.0]),
            ],
        )
        .await
        .unwrap();

    let writer_index = Arc::clone(&index);
    let writer = async move {
        for generation in 1..=50 {
            writer_index
                .replace_partition(
                    "active",
                    vec![
                        record(&format!("generation-{generation}:a"), &[1.0, 0.0]),
                        record(&format!("generation-{generation}:b"), &[0.0, 1.0]),
                    ],
                )
                .await
                .unwrap();
        }
    };

    let reader_index = Arc::clone(&index);
    let reader = async move {
        for _ in 0..200 {
            let result = reader_index
                .search(VectorSearchRequest::new(vec![1.0, 1.0], 2))
                .await
                .unwrap();
            assert_eq!(result.hits.len(), 2);
            assert_eq!(result.status.record_count, 2);
            let first_generation = result.hits[0].id.split(':').next().unwrap();
            let second_generation = result.hits[1].id.split(':').next().unwrap();
            assert_eq!(first_generation, second_generation);
        }
    };

    tokio::join!(writer, reader);
    assert_eq!(index.status().revision.value(), 51);
}

fn brute_force_cosine(
    query: &[f32],
    records: &[(String, Vec<f32>)],
    limit: usize,
) -> Vec<(String, f64)> {
    let query_norm = l2_norm(query);
    let mut scored = records
        .iter()
        .map(|(id, embedding)| {
            let score = embedding
                .iter()
                .zip(query)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>()
                / (l2_norm(embedding) * query_norm);
            (id.clone(), score)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(limit);
    scored
}

fn l2_norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_signed_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((self.0 >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32;
        unit * 2.0 - 1.0
    }
}
