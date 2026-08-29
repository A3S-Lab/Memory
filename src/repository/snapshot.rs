use super::validation::validate_count;
use super::{MemoryNamespace, MemoryNode, MemoryRepositoryError, MemoryStatus};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

/// Stable identity of the exact namespace-snapshot algorithm.
pub const MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1: &str = "a3s.memory.namespace-snapshot.sha256.v1";

/// Hard upper bound for one exact namespace snapshot.
pub const MAX_SNAPSHOT_NODES: usize = 100_000;

/// Hard upper bound for the canonical payload of one exact snapshot.
pub const MAX_SNAPSHOT_BYTES: usize = 256 * 1024 * 1024;

const SNAPSHOT_DIGEST_DOMAIN: &str = "a3s.memory.namespace-snapshot.v1";

/// Caller-selected scope and hard node/byte budgets for one exact repository view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySnapshotRequest {
    pub namespace: MemoryNamespace,
    pub statuses: BTreeSet<MemoryStatus>,
    pub max_nodes: usize,
    pub max_bytes: usize,
}

impl MemorySnapshotRequest {
    /// Select the complete current Active view of one exact namespace.
    pub fn new(namespace: MemoryNamespace, max_nodes: usize, max_bytes: usize) -> Self {
        Self {
            namespace,
            statuses: BTreeSet::from([MemoryStatus::Active]),
            max_nodes,
            max_bytes,
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
        )?;
        if self.max_bytes == 0 {
            return Err(MemoryRepositoryError::invalid(
                "snapshot.maxBytes",
                "must be greater than zero",
            ));
        }
        validate_count(
            "namespace snapshot bytes",
            self.max_bytes,
            MAX_SNAPSHOT_BYTES,
        )
    }
}

/// Complete, deterministically ordered current view selected by one request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNamespaceSnapshot {
    profile: String,
    namespace: MemoryNamespace,
    statuses: BTreeSet<MemoryStatus>,
    nodes: Vec<MemoryNode>,
    byte_count: usize,
    digest: String,
}

impl MemoryNamespaceSnapshot {
    /// Construct and hash a complete caller-provided view.
    ///
    /// Custom repositories should use this constructor rather than assembling
    /// response fields independently.
    pub fn try_new(
        request: MemorySnapshotRequest,
        nodes: Vec<MemoryNode>,
    ) -> Result<Self, MemoryRepositoryError> {
        build_snapshot(request, nodes)
    }

    /// Recompute and verify every response field against the original request.
    pub fn verify(&self, request: &MemorySnapshotRequest) -> Result<(), MemoryRepositoryError> {
        let expected = build_snapshot(request.clone(), self.nodes.clone())?;
        if expected != *self {
            return Err(MemoryRepositoryError::invariant(
                "namespace snapshot identity or shape does not match its request",
            ));
        }
        Ok(())
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn namespace(&self) -> &MemoryNamespace {
        &self.namespace
    }

    pub fn statuses(&self) -> &BTreeSet<MemoryStatus> {
        &self.statuses
    }

    pub fn nodes(&self) -> &[MemoryNode] {
        &self.nodes
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn byte_count(&self) -> usize {
        self.byte_count
    }

    pub fn into_nodes(self) -> Vec<MemoryNode> {
        self.nodes
    }
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
        .collect::<Vec<_>>();
    for node in &nodes {
        if node.namespace != request.namespace {
            return Err(MemoryRepositoryError::NamespaceMismatch {
                context: "namespace snapshot".into(),
            });
        }
    }
    let (digest, byte_count) = snapshot_identity(&request, &nodes)?;
    Ok(MemoryNamespaceSnapshot {
        profile: MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1.to_string(),
        namespace: request.namespace,
        statuses: request.statuses,
        nodes: nodes.into_iter().cloned().collect(),
        byte_count,
        digest,
    })
}

pub(crate) fn snapshot_from_nodes(
    request: MemorySnapshotRequest,
    nodes: Vec<MemoryNode>,
) -> Result<MemoryNamespaceSnapshot, MemoryRepositoryError> {
    MemoryNamespaceSnapshot::try_new(request, nodes)
}

fn build_snapshot(
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

    let (digest, byte_count) = snapshot_identity(&request, &nodes)?;
    Ok(MemoryNamespaceSnapshot {
        profile: MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1.to_string(),
        namespace: request.namespace,
        statuses: request.statuses,
        nodes,
        byte_count,
        digest,
    })
}

fn snapshot_identity<T: Serialize>(
    request: &MemorySnapshotRequest,
    nodes: &[T],
) -> Result<(String, usize), MemoryRepositoryError> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct DigestPayload<'a, T: Serialize> {
        profile: &'static str,
        namespace: &'a MemoryNamespace,
        statuses: &'a BTreeSet<MemoryStatus>,
        nodes: &'a [T],
    }

    let mut writer = BoundedDigestWriter::new(request.max_bytes);
    let result = serde_json::to_writer(
        &mut writer,
        &DigestPayload {
            profile: MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1,
            namespace: &request.namespace,
            statuses: &request.statuses,
            nodes,
        },
    );
    if writer.exceeded {
        return Err(MemoryRepositoryError::LimitExceeded {
            resource: "namespace snapshot bytes".into(),
            limit: request.max_bytes,
            actual: request.max_bytes.saturating_add(1),
        });
    }
    result.map_err(|error| {
        MemoryRepositoryError::invariant(format!(
            "namespace snapshot could not be encoded: {error}"
        ))
    })?;
    Ok((
        format!("sha256:{:x}", writer.hasher.finalize()),
        writer.byte_count,
    ))
}

struct BoundedDigestWriter {
    hasher: Sha256,
    byte_count: usize,
    limit: usize,
    exceeded: bool,
}

impl BoundedDigestWriter {
    fn new(limit: usize) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(SNAPSHOT_DIGEST_DOMAIN.as_bytes());
        hasher.update([0]);
        Self {
            hasher,
            byte_count: 0,
            limit,
            exceeded: false,
        }
    }
}

impl Write for BoundedDigestWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let next = self
            .byte_count
            .checked_add(buffer.len())
            .ok_or_else(|| std::io::Error::other("namespace snapshot byte count overflowed"))?;
        if next > self.limit {
            self.exceeded = true;
            return Err(std::io::Error::other(
                "namespace snapshot byte budget exceeded",
            ));
        }
        self.hasher.update(buffer);
        self.byte_count = next;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
