use super::validation::{validate_required_text, MAX_IDENTIFIER_BYTES};
use super::{MemoryNamespace, MemoryNode, MemoryRepositoryError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Explicit observation supplied after retrieval by an authorized host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryAccessEvent {
    pub id: String,
    pub namespace: MemoryNamespace,
    pub node_id: String,
    pub node_revision: u64,
    pub occurred_at: DateTime<Utc>,
    pub context_id: Option<String>,
}

impl MemoryAccessEvent {
    pub fn new(
        id: impl Into<String>,
        namespace: MemoryNamespace,
        node_id: impl Into<String>,
        node_revision: u64,
        occurred_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: id.into(),
            namespace,
            node_id: node_id.into(),
            node_revision,
            occurred_at,
            context_id: None,
        }
    }

    pub fn with_context_id(mut self, context_id: impl Into<String>) -> Self {
        self.context_id = Some(context_id.into());
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MemoryRepositoryError> {
        self.namespace.validate()?;
        validate_required_text("accessEvent.id", &self.id, MAX_IDENTIFIER_BYTES)?;
        validate_required_text("accessEvent.nodeId", &self.node_id, MAX_IDENTIFIER_BYTES)?;
        if self.node_revision == 0 {
            return Err(MemoryRepositoryError::invalid(
                "accessEvent.nodeRevision",
                "must be greater than zero",
            ));
        }
        if let Some(context_id) = &self.context_id {
            validate_required_text("accessEvent.contextId", context_id, MAX_IDENTIFIER_BYTES)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryUsageSummary {
    pub admissions: u64,
    pub uses: u64,
}

/// Deterministically ordered diagnostic snapshot of the reference repository.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryRepositorySnapshot {
    pub nodes: Vec<MemoryNode>,
    pub admissions: Vec<MemoryAccessEvent>,
    pub uses: Vec<MemoryAccessEvent>,
}
