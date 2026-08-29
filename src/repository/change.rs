use super::validation::{
    validate_count, validate_required_text, MAX_CHANGE_OPERATIONS, MAX_IDENTIFIER_BYTES,
};
use super::{
    EvidenceRef, MemoryNamespace, MemoryNode, MemoryNodeDraft, MemoryRelation,
    MemoryRepositoryError, MemoryStatus, RevisionMode,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One bounded mutation inside an atomic change set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum MemoryOperation {
    Create {
        node: MemoryNodeDraft,
    },
    Activate {
        node_id: String,
        expected_revision: u64,
    },
    Corroborate {
        node_id: String,
        expected_revision: u64,
        evidence: Vec<EvidenceRef>,
    },
    Revise {
        node_id: String,
        expected_revision: u64,
        content: String,
        mode: RevisionMode,
        evidence: Vec<EvidenceRef>,
        confidence: Option<f32>,
        importance: Option<f32>,
    },
    AddRelation {
        node_id: String,
        expected_revision: u64,
        relation: MemoryRelation,
    },
    RemoveRelation {
        node_id: String,
        expected_revision: u64,
        relation: MemoryRelation,
    },
    SetStatus {
        node_id: String,
        expected_revision: u64,
        status: MemoryStatus,
    },
}

/// Caller-timestamped, idempotent unit of atomic mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryChangeSet {
    pub idempotency_key: String,
    pub namespace: MemoryNamespace,
    pub occurred_at: DateTime<Utc>,
    pub operations: Vec<MemoryOperation>,
}

impl MemoryChangeSet {
    pub fn new(
        idempotency_key: impl Into<String>,
        namespace: MemoryNamespace,
        occurred_at: DateTime<Utc>,
        operations: Vec<MemoryOperation>,
    ) -> Self {
        Self {
            idempotency_key: idempotency_key.into(),
            namespace,
            occurred_at,
            operations,
        }
    }

    pub(crate) fn validate_shape(&self) -> Result<(), MemoryRepositoryError> {
        self.namespace.validate()?;
        validate_required_text(
            "changeSet.idempotencyKey",
            &self.idempotency_key,
            MAX_IDENTIFIER_BYTES,
        )?;
        if self.operations.is_empty() {
            return Err(MemoryRepositoryError::invalid(
                "changeSet.operations",
                "must contain at least one operation",
            ));
        }
        validate_count(
            "change-set operations",
            self.operations.len(),
            MAX_CHANGE_OPERATIONS,
        )
    }
}

/// Deterministic result retained for idempotent replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryChangeResult {
    pub idempotency_key: String,
    pub occurred_at: DateTime<Utc>,
    pub nodes: Vec<MemoryNode>,
}
