use super::change_engine::{stage_change_set, StagedChange};
use super::query::query_nodes;
use super::validation::{validate_required_text, MAX_IDENTIFIER_BYTES};
use super::{
    MemoryAccessEvent, MemoryChangeResult, MemoryChangeSet, MemoryNamespace, MemoryNode,
    MemoryQuery, MemoryQueryResult, MemoryRepository, MemoryRepositoryError,
    MemoryRepositorySnapshot, MemoryUsageSummary,
};
use std::collections::BTreeMap;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
struct AppliedChange {
    change_set: MemoryChangeSet,
    result: MemoryChangeResult,
}

#[derive(Debug, Default)]
struct RepositoryState {
    nodes: BTreeMap<MemoryNamespace, BTreeMap<String, MemoryNode>>,
    applied_changes: BTreeMap<MemoryNamespace, BTreeMap<String, AppliedChange>>,
    admissions: BTreeMap<MemoryNamespace, BTreeMap<String, MemoryAccessEvent>>,
    uses: BTreeMap<MemoryNamespace, BTreeMap<String, MemoryAccessEvent>>,
    usage: BTreeMap<MemoryNamespace, BTreeMap<String, MemoryUsageSummary>>,
}

/// In-memory executable reference implementation of [`MemoryRepository`].
#[derive(Debug, Default)]
pub struct InMemoryRepository {
    state: RwLock<RepositoryState>,
}

impl InMemoryRepository {
    pub fn new() -> Self {
        Self::default()
    }

    /// Capture deterministic visible state for conformance and diagnostics.
    pub async fn snapshot(&self) -> MemoryRepositorySnapshot {
        let state = self.state.read().await;
        MemoryRepositorySnapshot {
            nodes: state
                .nodes
                .values()
                .flat_map(BTreeMap::values)
                .cloned()
                .collect(),
            admissions: state
                .admissions
                .values()
                .flat_map(BTreeMap::values)
                .cloned()
                .collect(),
            uses: state
                .uses
                .values()
                .flat_map(BTreeMap::values)
                .cloned()
                .collect(),
        }
    }

    pub(crate) async fn preview_apply(
        &self,
        change_set: &MemoryChangeSet,
    ) -> Result<(MemoryChangeResult, bool), MemoryRepositoryError> {
        let state = self.state.read().await;
        let prepared = prepare_change(&state, change_set)?;
        Ok((prepared.result, prepared.staged.is_none()))
    }

    pub(crate) async fn preview_admission(
        &self,
        event: &MemoryAccessEvent,
    ) -> Result<bool, MemoryRepositoryError> {
        self.preview_access(event, AccessKind::Admission).await
    }

    pub(crate) async fn preview_use(
        &self,
        event: &MemoryAccessEvent,
    ) -> Result<bool, MemoryRepositoryError> {
        self.preview_access(event, AccessKind::Use).await
    }

    async fn preview_access(
        &self,
        event: &MemoryAccessEvent,
        kind: AccessKind,
    ) -> Result<bool, MemoryRepositoryError> {
        event.validate()?;
        let state = self.state.read().await;
        validate_access(&state, event, kind)
    }

    async fn record_access(
        &self,
        event: MemoryAccessEvent,
        kind: AccessKind,
    ) -> Result<(), MemoryRepositoryError> {
        event.validate()?;
        let mut state = self.state.write().await;
        if validate_access(&state, &event, kind)? {
            return Ok(());
        }

        let current_summary = state
            .usage
            .get(&event.namespace)
            .and_then(|nodes| nodes.get(&event.node_id))
            .copied()
            .unwrap_or_default();
        let mut next_summary = current_summary;
        match kind {
            AccessKind::Admission => {
                next_summary.admissions =
                    next_summary.admissions.checked_add(1).ok_or_else(|| {
                        MemoryRepositoryError::invariant("admission counter overflow")
                    })?;
            }
            AccessKind::Use => {
                next_summary.uses = next_summary
                    .uses
                    .checked_add(1)
                    .ok_or_else(|| MemoryRepositoryError::invariant("use counter overflow"))?;
            }
        }

        match kind {
            AccessKind::Admission => &mut state.admissions,
            AccessKind::Use => &mut state.uses,
        }
        .entry(event.namespace.clone())
        .or_default()
        .insert(event.id.clone(), event.clone());
        state
            .usage
            .entry(event.namespace)
            .or_default()
            .insert(event.node_id, next_summary);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum AccessKind {
    Admission,
    Use,
}

#[async_trait::async_trait]
impl MemoryRepository for InMemoryRepository {
    async fn apply(
        &self,
        change_set: MemoryChangeSet,
    ) -> Result<MemoryChangeResult, MemoryRepositoryError> {
        let mut state = self.state.write().await;
        let PreparedChange { result, staged } = prepare_change(&state, &change_set)?;
        let Some(mut staged) = staged else {
            return Ok(result);
        };
        let namespace_nodes = state.nodes.entry(change_set.namespace.clone()).or_default();
        for node_id in &staged.changed {
            let node = staged.nodes.remove(node_id).ok_or_else(|| {
                MemoryRepositoryError::invariant(format!(
                    "changed memory node {node_id} was not staged"
                ))
            })?;
            namespace_nodes.insert(node_id.clone(), node);
        }
        state
            .applied_changes
            .entry(change_set.namespace.clone())
            .or_default()
            .insert(
                change_set.idempotency_key.clone(),
                AppliedChange {
                    change_set,
                    result: result.clone(),
                },
            );
        Ok(result)
    }

    async fn get(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<Option<MemoryNode>, MemoryRepositoryError> {
        namespace.validate()?;
        validate_required_text("nodeId", node_id, MAX_IDENTIFIER_BYTES)?;
        let state = self.state.read().await;
        Ok(state
            .nodes
            .get(namespace)
            .and_then(|nodes| nodes.get(node_id))
            .cloned())
    }

    async fn query(&self, query: MemoryQuery) -> Result<MemoryQueryResult, MemoryRepositoryError> {
        query.validate()?;
        let state = self.state.read().await;
        Ok(query_nodes(state.nodes.get(&query.namespace), &query))
    }

    async fn record_admission(
        &self,
        event: MemoryAccessEvent,
    ) -> Result<(), MemoryRepositoryError> {
        self.record_access(event, AccessKind::Admission).await
    }

    async fn record_use(&self, event: MemoryAccessEvent) -> Result<(), MemoryRepositoryError> {
        self.record_access(event, AccessKind::Use).await
    }

    async fn usage_summary(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<MemoryUsageSummary, MemoryRepositoryError> {
        namespace.validate()?;
        validate_required_text("nodeId", node_id, MAX_IDENTIFIER_BYTES)?;
        let state = self.state.read().await;
        if state
            .nodes
            .get(namespace)
            .and_then(|nodes| nodes.get(node_id))
            .is_none()
        {
            return Err(MemoryRepositoryError::NodeNotFound {
                node_id: node_id.to_owned(),
            });
        }
        Ok(state
            .usage
            .get(namespace)
            .and_then(|nodes| nodes.get(node_id))
            .copied()
            .unwrap_or_default())
    }
}

struct PreparedChange {
    result: MemoryChangeResult,
    staged: Option<StagedChange>,
}

fn prepare_change(
    state: &RepositoryState,
    change_set: &MemoryChangeSet,
) -> Result<PreparedChange, MemoryRepositoryError> {
    change_set.validate_shape()?;
    if let Some(applied) = state
        .applied_changes
        .get(&change_set.namespace)
        .and_then(|changes| changes.get(&change_set.idempotency_key))
    {
        return if applied.change_set == *change_set {
            Ok(PreparedChange {
                result: applied.result.clone(),
                staged: None,
            })
        } else {
            Err(MemoryRepositoryError::IdempotencyConflict {
                key: change_set.idempotency_key.clone(),
            })
        };
    }

    let staged = stage_change_set(change_set, state.nodes.get(&change_set.namespace))?;
    let result = MemoryChangeResult {
        idempotency_key: change_set.idempotency_key.clone(),
        occurred_at: change_set.occurred_at,
        nodes: staged
            .changed
            .iter()
            .filter_map(|node_id| staged.nodes.get(node_id).cloned())
            .collect(),
    };
    Ok(PreparedChange {
        result,
        staged: Some(staged),
    })
}

fn validate_access(
    state: &RepositoryState,
    event: &MemoryAccessEvent,
    kind: AccessKind,
) -> Result<bool, MemoryRepositoryError> {
    let node = state
        .nodes
        .get(&event.namespace)
        .and_then(|nodes| nodes.get(&event.node_id))
        .ok_or_else(|| MemoryRepositoryError::NodeNotFound {
            node_id: event.node_id.clone(),
        })?;
    let revision_updated_at = node
        .revision_updated_at(event.node_revision)
        .ok_or_else(|| MemoryRepositoryError::NodeRevisionNotFound {
            node_id: event.node_id.clone(),
            revision: event.node_revision,
        })?;
    if matches!(kind, AccessKind::Admission)
        && (node.revision != event.node_revision || node.status != super::MemoryStatus::Active)
    {
        return Err(MemoryRepositoryError::AdmissionNotAllowed {
            node_id: event.node_id.clone(),
            revision: event.node_revision,
            current_revision: node.revision,
            current_status: node.status.as_str().into(),
        });
    }
    if event.occurred_at < revision_updated_at {
        return Err(MemoryRepositoryError::invalid(
            "accessEvent.occurredAt",
            "must not precede the referenced node revision",
        ));
    }

    let existing = match kind {
        AccessKind::Admission => state
            .admissions
            .get(&event.namespace)
            .and_then(|events| events.get(&event.id)),
        AccessKind::Use => state
            .uses
            .get(&event.namespace)
            .and_then(|events| events.get(&event.id)),
    };
    match existing {
        Some(existing) if existing == event => Ok(true),
        Some(_) => Err(MemoryRepositoryError::IdempotencyConflict {
            key: event.id.clone(),
        }),
        None => Ok(false),
    }
}
