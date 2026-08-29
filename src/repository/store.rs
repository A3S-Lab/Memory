use super::{
    MemoryAccessEvent, MemoryChangeResult, MemoryChangeSet, MemoryNamespace, MemoryNode,
    MemoryQuery, MemoryQueryResult, MemoryRepositoryError, MemoryUsageSummary,
};

/// Repository contract for evidence-backed durable memory.
#[async_trait::async_trait]
pub trait MemoryRepository: Send + Sync {
    /// Atomically apply a bounded, revision-checked change set.
    async fn apply(
        &self,
        change_set: MemoryChangeSet,
    ) -> Result<MemoryChangeResult, MemoryRepositoryError>;

    /// Read one node from an exact namespace without recording access.
    async fn get(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<Option<MemoryNode>, MemoryRepositoryError>;

    /// Query one exact namespace without mutating repository state.
    async fn query(&self, query: MemoryQuery) -> Result<MemoryQueryResult, MemoryRepositoryError>;

    /// Record that the host admitted a node into a model context.
    async fn record_admission(&self, event: MemoryAccessEvent)
        -> Result<(), MemoryRepositoryError>;

    /// Record that the host cited, selected, or otherwise used a node.
    async fn record_use(&self, event: MemoryAccessEvent) -> Result<(), MemoryRepositoryError>;

    /// Return explicit admission and use counts for a node.
    async fn usage_summary(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<MemoryUsageSummary, MemoryRepositoryError>;
}
