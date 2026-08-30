use super::{
    MemoryAccessEvent, MemoryChangeResult, MemoryChangeSet, MemoryNamespace,
    MemoryNamespaceChangeToken, MemoryNamespaceSnapshot, MemoryNode, MemoryQuery,
    MemoryQueryResult, MemoryRepositoryError, MemorySnapshotRequest, MemoryUsageSummary,
    MAX_QUERY_LIMIT,
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

    /// Capture one complete, bounded, deterministic namespace view.
    ///
    /// The default implementation is exact below the ordinary query horizon.
    /// Backends that can atomically enumerate larger views should override it.
    async fn snapshot_namespace(
        &self,
        request: MemorySnapshotRequest,
    ) -> Result<MemoryNamespaceSnapshot, MemoryRepositoryError> {
        request.validate()?;
        let query_limit = request.max_nodes.saturating_add(1).min(MAX_QUERY_LIMIT);
        let result = self
            .query(
                MemoryQuery::new(request.namespace.clone())
                    .with_statuses(request.statuses.iter().copied())
                    .with_limit(query_limit),
            )
            .await?;
        if request.max_nodes >= MAX_QUERY_LIMIT && result.hits.len() == MAX_QUERY_LIMIT {
            return Err(MemoryRepositoryError::LimitExceeded {
                resource: "namespace snapshot query horizon".into(),
                limit: MAX_QUERY_LIMIT - 1,
                actual: MAX_QUERY_LIMIT,
            });
        }
        super::snapshot::snapshot_from_nodes(
            request,
            result.hits.into_iter().map(|hit| hit.node).collect(),
        )
    }

    /// Return an optional exact-namespace change token.
    ///
    /// Some opts into the MemoryNamespaceChangeToken contract: every novel
    /// successful apply that changes node state in this namespace must publish
    /// a different token at the same linearization point. Tokens must remain
    /// stable across reads, idempotent replay, access events, and durable
    /// restart. Backends that cannot make those guarantees return None.
    async fn namespace_change_token(
        &self,
        namespace: &MemoryNamespace,
    ) -> Result<Option<MemoryNamespaceChangeToken>, MemoryRepositoryError> {
        namespace.validate()?;
        Ok(None)
    }

    /// Record that the host admitted the current active node revision into a model context.
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
