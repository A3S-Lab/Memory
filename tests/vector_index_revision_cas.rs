use a3s_memory::{
    InMemoryVectorIndex, VectorIndex, VectorIndexDescriptor, VectorIndexError, VectorIndexStatus,
    VectorMutationConsistency, VectorRecord, VectorResult, VectorRevision, VectorSearchRequest,
    VectorSearchResult,
};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Barrier;

fn record(id: &str, embedding: &[f32]) -> VectorRecord {
    VectorRecord::new(id, embedding.to_vec())
}

struct PartitionAtomicOnlyIndex {
    inner: InMemoryVectorIndex,
}

#[async_trait]
impl VectorIndex for PartitionAtomicOnlyIndex {
    fn descriptor(&self) -> &VectorIndexDescriptor {
        self.inner.descriptor()
    }

    fn status(&self) -> VectorIndexStatus {
        self.inner.status()
    }

    async fn replace_partition(
        &self,
        partition: &str,
        records: Vec<VectorRecord>,
    ) -> VectorResult<VectorIndexStatus> {
        self.inner.replace_partition(partition, records).await
    }

    async fn remove_partition(&self, partition: &str) -> VectorResult<VectorIndexStatus> {
        self.inner.remove_partition(partition).await
    }

    async fn search(&self, request: VectorSearchRequest) -> VectorResult<VectorSearchResult> {
        self.inner.search(request).await
    }

    async fn clear(&self) -> VectorResult<VectorIndexStatus> {
        self.inner.clear().await
    }
}

#[tokio::test]
async fn custom_backend_defaults_remain_source_compatible_and_fail_closed_for_cas() {
    let index = PartitionAtomicOnlyIndex {
        inner: InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap(),
    };
    assert_eq!(
        index.mutation_consistency(),
        VectorMutationConsistency::PartitionAtomic
    );
    let observation = index.observe().await.unwrap();
    assert_eq!(observation.status, VectorIndexStatus::default());
    assert!(observation.change_token.is_none());

    let error = index
        .replace_partition_if_revision(
            "semantic",
            VectorRevision::new(0),
            vec![record("unpublished", &[1.0, 0.0])],
        )
        .await
        .unwrap_err();

    assert_eq!(error, VectorIndexError::ConditionalMutationUnsupported);
    assert_eq!(index.status(), VectorIndexStatus::default());
}

#[tokio::test]
async fn stale_conditional_replacement_cannot_overwrite_a_newer_revision() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    assert_eq!(
        index.mutation_consistency(),
        VectorMutationConsistency::IndexRevisionCas
    );

    let published = index
        .replace_partition_if_revision(
            "semantic",
            VectorRevision::new(0),
            vec![record("current", &[1.0, 0.0])],
        )
        .await
        .unwrap();
    assert_eq!(published.revision, VectorRevision::new(1));

    let error = index
        .replace_partition_if_revision(
            "semantic",
            VectorRevision::new(0),
            vec![record("stale", &[0.0, 1.0])],
        )
        .await
        .unwrap_err();
    assert_eq!(
        error,
        VectorIndexError::RevisionConflict {
            expected: VectorRevision::new(0),
            actual: VectorRevision::new(1),
        }
    );

    let result = index
        .search(VectorSearchRequest::new(vec![1.0, 0.0], 10).with_partition("semantic"))
        .await
        .unwrap();
    assert_eq!(result.status, published);
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, "current");
}

#[tokio::test]
async fn stale_conditional_cleanup_cannot_remove_a_newer_partition() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    let first = index
        .replace_partition_if_revision(
            "semantic",
            VectorRevision::new(0),
            vec![record("old", &[1.0, 0.0])],
        )
        .await
        .unwrap();
    let newer = index
        .replace_partition_if_revision("semantic", first.revision, vec![record("new", &[0.0, 1.0])])
        .await
        .unwrap();

    let error = index
        .remove_partition_if_revision("semantic", first.revision)
        .await
        .unwrap_err();
    assert_eq!(
        error,
        VectorIndexError::RevisionConflict {
            expected: first.revision,
            actual: newer.revision,
        }
    );

    let result = index
        .search(VectorSearchRequest::new(vec![0.0, 1.0], 10).with_partition("semantic"))
        .await
        .unwrap();
    assert_eq!(result.status, newer);
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, "new");
}

#[tokio::test]
async fn concurrent_compare_and_swap_has_exactly_one_winner() {
    let index = Arc::new(InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap());
    let barrier = Arc::new(Barrier::new(3));
    let mut tasks = Vec::new();
    for (id, embedding) in [("left", [1.0, 0.0]), ("right", [0.0, 1.0])] {
        let index = Arc::clone(&index);
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            index
                .replace_partition_if_revision(
                    "semantic",
                    VectorRevision::new(0),
                    vec![record(id, &embedding)],
                )
                .await
        }));
    }
    barrier.wait().await;

    let mut success_count = 0;
    let mut conflict_count = 0;
    for task in tasks {
        match task.await.unwrap() {
            Ok(status) => {
                success_count += 1;
                assert_eq!(status.revision, VectorRevision::new(1));
            }
            Err(VectorIndexError::RevisionConflict { expected, actual }) => {
                conflict_count += 1;
                assert_eq!(expected, VectorRevision::new(0));
                assert_eq!(actual, VectorRevision::new(1));
            }
            Err(error) => panic!("unexpected conditional mutation error: {error}"),
        }
    }

    assert_eq!(success_count, 1);
    assert_eq!(conflict_count, 1);
    assert_eq!(index.status().revision, VectorRevision::new(1));
    assert_eq!(index.status().record_count, 1);
}
