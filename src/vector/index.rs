use super::{
    VectorIndexDescriptor, VectorIndexStatus, VectorRecord, VectorResult, VectorSearchRequest,
    VectorSearchResult,
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

    /// Atomically replace every record in `partition`.
    ///
    /// Replacing an existing partition with an empty record list removes it.
    /// Replacing a missing partition with an empty list is a no-op.
    async fn replace_partition(
        &self,
        partition: &str,
        records: Vec<VectorRecord>,
    ) -> VectorResult<VectorIndexStatus>;

    /// Atomically remove one partition. Missing partitions are a no-op.
    async fn remove_partition(&self, partition: &str) -> VectorResult<VectorIndexStatus>;

    /// Search one immutable index revision.
    async fn search(&self, request: VectorSearchRequest) -> VectorResult<VectorSearchResult>;

    /// Remove every partition. Clearing an empty index is a no-op.
    async fn clear(&self) -> VectorResult<VectorIndexStatus>;
}
