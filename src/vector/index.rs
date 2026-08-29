use super::{
    VectorIndexDescriptor, VectorIndexError, VectorIndexStatus, VectorMutationConsistency,
    VectorRecord, VectorResult, VectorRevision, VectorSearchRequest, VectorSearchResult,
};

/// A bounded vector index whose content and lifecycle are owned by its caller.
///
/// Partitions are the atomic mutation unit. Implementations must make a
/// successful replacement visible in one revision and must not expose a
/// partially constructed partition to concurrent searches.
#[async_trait::async_trait]
pub trait VectorIndex: Send + Sync {
    /// Return the immutable shape and resource limits of this index.
    fn descriptor(&self) -> &VectorIndexDescriptor;

    /// Return the latest published status without waiting for background work.
    fn status(&self) -> VectorIndexStatus;

    /// Return the strongest partition-mutation ordering contract implemented
    /// by this backend.
    fn mutation_consistency(&self) -> VectorMutationConsistency {
        VectorMutationConsistency::PartitionAtomic
    }

    /// Atomically replace every record in `partition`.
    ///
    /// Replacing an existing partition with an empty record list removes it.
    /// Replacing a missing partition with an empty list is a no-op.
    async fn replace_partition(
        &self,
        partition: &str,
        records: Vec<VectorRecord>,
    ) -> VectorResult<VectorIndexStatus>;

    /// Atomically replace one partition only when the complete index still has
    /// `expected_revision`.
    ///
    /// Implementations advertising `IndexRevisionCas` must compare and mutate
    /// at one linearization point. The default fails closed so a custom backend
    /// cannot accidentally claim cross-writer ordering from a check-then-write.
    async fn replace_partition_if_revision(
        &self,
        _partition: &str,
        _expected_revision: VectorRevision,
        _records: Vec<VectorRecord>,
    ) -> VectorResult<VectorIndexStatus> {
        Err(VectorIndexError::ConditionalMutationUnsupported)
    }

    /// Atomically remove one partition. Missing partitions are a no-op.
    async fn remove_partition(&self, partition: &str) -> VectorResult<VectorIndexStatus>;

    /// Atomically remove one partition only when the complete index still has
    /// `expected_revision`.
    async fn remove_partition_if_revision(
        &self,
        _partition: &str,
        _expected_revision: VectorRevision,
    ) -> VectorResult<VectorIndexStatus> {
        Err(VectorIndexError::ConditionalMutationUnsupported)
    }

    /// Search one immutable index revision.
    async fn search(&self, request: VectorSearchRequest) -> VectorResult<VectorSearchResult>;

    /// Remove every partition. Clearing an empty index is a no-op.
    async fn clear(&self) -> VectorResult<VectorIndexStatus>;
}
