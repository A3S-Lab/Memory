use super::validation::validate_count;
use super::{MemoryNamespace, MemoryNode, MemoryRepositoryError, MemoryStatus};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Stable identity of the exact namespace-snapshot algorithm.
pub const MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1: &str = "a3s.memory.namespace-snapshot.sha256.v1";

/// Hard upper bound for one exact namespace snapshot.
pub const MAX_SNAPSHOT_NODES: usize = 100_000;

const SNAPSHOT_DIGEST_DOMAIN: &str = "a3s.memory.namespace-snapshot.v1";

/// Caller-selected scope and hard node budget for one exact repository view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySnapshotRequest {
    pub namespace: MemoryNamespace,
    pub statuses: BTreeSet<MemoryStatus>,
    pub max_nodes: usize,
}

impl MemorySnapshotRequest {
    /// Select the complete current Active view of one exact namespace.
    pub fn new(namespace: MemoryNamespace, max_nodes: usize) -> Self {
        Self {
            namespace,
            statuses: BTreeSet::from([MemoryStatus::Active]),
            max_nodes,
        }
    }

    pub fn with_statuses(mut self, statuses: impl IntoIterator<Item = MemoryStatus>) -> Self {
        self.statuses = statuses.into_iter().collect();
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MemoryRepositoryError> {
        self.namespace.validate()?;
        if self.statuses.is_empty() {
            return Err(MemoryRepositoryError::invalid(
                "snapshot.statuses",
                "must contain at least one status",
            ));
        }
        if self.max_nodes == 0 {
            return Err(MemoryRepositoryError::invalid(
                "snapshot.maxNodes",
                "must be greater than zero",
            ));
        }
        validate_count(
            "namespace snapshot nodes",
            self.max_nodes,
            MAX_SNAPSHOT_NODES,
        )
    }
}

/// Complete, deterministically ordered current view selected by one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNamespaceSnapshot {
    pub profile: String,
    pub namespace: MemoryNamespace,
    pub statuses: BTreeSet<MemoryStatus>,
    pub nodes: Vec<MemoryNode>,
    pub digest: String,
}

pub(crate) fn snapshot_from_map(
    namespace_nodes: Option<&BTreeMap<String, MemoryNode>>,
    request: MemorySnapshotRequest,
) -> Result<MemoryNamespaceSnapshot, MemoryRepositoryError> {
    request.validate()?;
    let actual = namespace_nodes
        .into_iter()
        .flat_map(BTreeMap::values)
        .filter(|node| request.statuses.contains(&node.status))
        .count();
    if actual > request.max_nodes {
        return Err(MemoryRepositoryError::LimitExceeded {
            resource: "namespace snapshot nodes".into(),
            limit: request.max_nodes,
            actual,
        });
    }
    let nodes = namespace_nodes
        .into_iter()
        .flat_map(BTreeMap::values)
        .filter(|node| request.statuses.contains(&node.status))
        .cloned()
        .collect();
    snapshot_from_nodes(request, nodes)
}

pub(crate) fn snapshot_from_nodes(
    request: MemorySnapshotRequest,
    mut nodes: Vec<MemoryNode>,
) -> Result<MemoryNamespaceSnapshot, MemoryRepositoryError> {
    request.validate()?;
    nodes.sort_by(|left, right| left.id.cmp(&right.id));
    if nodes.len() > request.max_nodes {
        return Err(MemoryRepositoryError::LimitExceeded {
            resource: "namespace snapshot nodes".into(),
            limit: request.max_nodes,
            actual: nodes.len(),
        });
    }
    let mut previous_id: Option<&str> = None;
    for node in &nodes {
        if node.namespace != request.namespace || !request.statuses.contains(&node.status) {
            return Err(MemoryRepositoryError::NamespaceMismatch {
                context: "namespace snapshot".into(),
            });
        }
        if previous_id == Some(node.id.as_str()) {
            return Err(MemoryRepositoryError::invariant(format!(
                "namespace snapshot contains duplicate node {}",
                node.id
            )));
        }
        previous_id = Some(&node.id);
    }

    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct DigestPayload<'a> {
        profile: &'static str,
        namespace: &'a MemoryNamespace,
        statuses: &'a BTreeSet<MemoryStatus>,
        nodes: &'a [MemoryNode],
    }

    let encoded = serde_json::to_vec(&DigestPayload {
        profile: MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1,
        namespace: &request.namespace,
        statuses: &request.statuses,
        nodes: &nodes,
    })
    .map_err(|error| {
        MemoryRepositoryError::invariant(format!(
            "namespace snapshot could not be encoded: {error}"
        ))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(SNAPSHOT_DIGEST_DOMAIN.as_bytes());
    hasher.update([0]);
    hasher.update(encoded);
    Ok(MemoryNamespaceSnapshot {
        profile: MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1.to_string(),
        namespace: request.namespace,
        statuses: request.statuses,
        nodes,
        digest: format!("sha256:{:x}", hasher.finalize()),
    })
}
