use super::validation::{
    validate_count, validate_optional_text, validate_required_text, validate_unit_float,
    MAX_CONTENT_BYTES, MAX_DIGEST_BYTES, MAX_EVIDENCE_PER_NODE, MAX_IDENTIFIER_BYTES,
    MAX_LABELS_PER_NODE, MAX_LABEL_KEY_BYTES, MAX_LABEL_VALUE_BYTES, MAX_RELATIONS_PER_NODE,
    MAX_REVISIONS_PER_NODE, MAX_URI_BYTES,
};
use super::MemoryRepositoryError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Exact tenant, principal, and scope partition for repository operations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNamespace {
    tenant_id: String,
    principal_id: String,
    scope_id: String,
}

impl MemoryNamespace {
    pub fn try_new(
        tenant_id: impl Into<String>,
        principal_id: impl Into<String>,
        scope_id: impl Into<String>,
    ) -> Result<Self, MemoryRepositoryError> {
        let namespace = Self {
            tenant_id: tenant_id.into(),
            principal_id: principal_id.into(),
            scope_id: scope_id.into(),
        };
        namespace.validate()?;
        Ok(namespace)
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    pub fn principal_id(&self) -> &str {
        &self.principal_id
    }

    pub fn scope_id(&self) -> &str {
        &self.scope_id
    }

    pub(crate) fn validate(&self) -> Result<(), MemoryRepositoryError> {
        validate_required_text("namespace.tenantId", &self.tenant_id, MAX_IDENTIFIER_BYTES)?;
        validate_required_text(
            "namespace.principalId",
            &self.principal_id,
            MAX_IDENTIFIER_BYTES,
        )?;
        validate_required_text("namespace.scopeId", &self.scope_id, MAX_IDENTIFIER_BYTES)
    }
}

/// Source class for immutable evidence referenced by a memory node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    UserStatement,
    ToolResult,
    Artifact,
    SessionTurn,
    Verification,
    Manual,
}

/// Reference to immutable source evidence owned by the host runtime.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRef {
    pub uri: String,
    pub digest: String,
    pub kind: EvidenceKind,
    pub occurred_at: DateTime<Utc>,
}

impl EvidenceRef {
    pub fn try_new(
        uri: impl Into<String>,
        digest: impl Into<String>,
        kind: EvidenceKind,
        occurred_at: DateTime<Utc>,
    ) -> Result<Self, MemoryRepositoryError> {
        let evidence = Self {
            uri: uri.into(),
            digest: digest.into(),
            kind,
            occurred_at,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub(crate) fn validate(&self) -> Result<(), MemoryRepositoryError> {
        validate_required_text("evidence.uri", &self.uri, MAX_URI_BYTES)?;
        validate_required_text("evidence.digest", &self.digest, MAX_DIGEST_BYTES)
    }
}

/// Durable memory class. Working memory remains host runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurableMemoryKind {
    Episodic,
    Semantic,
    Procedural,
}

/// Lifecycle state for a durable memory node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Candidate,
    Active,
    Superseded,
    Conflicted,
    Tombstoned,
}

impl MemoryStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Conflicted => "conflicted",
            Self::Tombstoned => "tombstoned",
        }
    }
}

/// Typed relationship between nodes in the same exact namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRelationKind {
    Supersedes,
    SupersededBy,
    ConflictsWith,
    RelatedTo,
}

impl MemoryRelationKind {
    pub(crate) fn inverse(self) -> Self {
        match self {
            Self::Supersedes => Self::SupersededBy,
            Self::SupersededBy => Self::Supersedes,
            Self::ConflictsWith => Self::ConflictsWith,
            Self::RelatedTo => Self::RelatedTo,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryRelation {
    pub kind: MemoryRelationKind,
    pub target_id: String,
}

impl MemoryRelation {
    pub fn new(kind: MemoryRelationKind, target_id: impl Into<String>) -> Self {
        Self {
            kind,
            target_id: target_id.into(),
        }
    }

    pub(crate) fn validate(&self, node_id: &str) -> Result<(), MemoryRepositoryError> {
        validate_required_text("relation.targetId", &self.target_id, MAX_IDENTIFIER_BYTES)?;
        if self.target_id == node_id {
            return Err(MemoryRepositoryError::invariant(format!(
                "memory node {node_id} cannot relate to itself"
            )));
        }
        Ok(())
    }
}

/// Semantic intent of a content revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionMode {
    Refinement,
    Correction,
}

/// Audit classification for the revision currently represented by a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRevisionKind {
    Created,
    Activated,
    Corroborated,
    Refined,
    Corrected,
    RelationChanged,
    StatusChanged,
}

/// Immutable snapshot of a prior node revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNodeRevision {
    pub revision: u64,
    pub kind: DurableMemoryKind,
    pub status: MemoryStatus,
    pub content: String,
    pub confidence: f32,
    pub importance: f32,
    pub evidence: Vec<EvidenceRef>,
    pub relations: Vec<MemoryRelation>,
    pub labels: BTreeMap<String, String>,
    pub updated_at: DateTime<Utc>,
    pub revision_kind: MemoryRevisionKind,
}

/// Current immutable snapshot plus the complete prior revision history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNode {
    pub id: String,
    pub namespace: MemoryNamespace,
    pub revision: u64,
    pub kind: DurableMemoryKind,
    pub status: MemoryStatus,
    pub content: String,
    pub confidence: f32,
    pub importance: f32,
    pub evidence: Vec<EvidenceRef>,
    pub relations: Vec<MemoryRelation>,
    pub labels: BTreeMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub revision_kind: MemoryRevisionKind,
    pub history: Vec<MemoryNodeRevision>,
}

impl MemoryNode {
    pub(crate) fn from_draft(draft: MemoryNodeDraft) -> Self {
        Self {
            id: draft.id,
            namespace: draft.namespace,
            revision: 1,
            kind: draft.kind,
            status: draft.status,
            content: draft.content,
            confidence: draft.confidence,
            importance: draft.importance,
            evidence: draft.evidence,
            relations: draft.relations,
            labels: draft.labels,
            created_at: draft.created_at,
            updated_at: draft.created_at,
            revision_kind: MemoryRevisionKind::Created,
            history: Vec::new(),
        }
    }

    pub(crate) fn begin_revision(
        &mut self,
        updated_at: DateTime<Utc>,
        revision_kind: MemoryRevisionKind,
    ) -> Result<(), MemoryRepositoryError> {
        validate_count(
            "node revisions",
            self.history.len() + 1,
            MAX_REVISIONS_PER_NODE,
        )?;
        if updated_at < self.updated_at {
            return Err(MemoryRepositoryError::invalid(
                "changeSet.occurredAt",
                "must not precede the node's current update time",
            ));
        }
        self.history.push(MemoryNodeRevision {
            revision: self.revision,
            kind: self.kind,
            status: self.status,
            content: self.content.clone(),
            confidence: self.confidence,
            importance: self.importance,
            evidence: self.evidence.clone(),
            relations: self.relations.clone(),
            labels: self.labels.clone(),
            updated_at: self.updated_at,
            revision_kind: self.revision_kind,
        });
        self.revision += 1;
        self.updated_at = updated_at;
        self.revision_kind = revision_kind;
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), MemoryRepositoryError> {
        self.namespace.validate()?;
        validate_required_text("node.id", &self.id, MAX_IDENTIFIER_BYTES)?;
        validate_required_text("node.content", &self.content, MAX_CONTENT_BYTES)?;
        validate_unit_float("node.confidence", self.confidence)?;
        validate_unit_float("node.importance", self.importance)?;
        validate_count("node evidence", self.evidence.len(), MAX_EVIDENCE_PER_NODE)?;
        validate_count(
            "node relations",
            self.relations.len(),
            MAX_RELATIONS_PER_NODE,
        )?;
        validate_count("node labels", self.labels.len(), MAX_LABELS_PER_NODE)?;
        validate_count("node revisions", self.history.len(), MAX_REVISIONS_PER_NODE)?;
        if self.created_at > self.updated_at {
            return Err(MemoryRepositoryError::invalid(
                "node.createdAt",
                "must not follow updatedAt",
            ));
        }
        if self.status == MemoryStatus::Active && self.evidence.is_empty() {
            return Err(MemoryRepositoryError::EvidenceRequired {
                node_id: self.id.clone(),
            });
        }
        for item in &self.evidence {
            item.validate()?;
        }
        if self
            .evidence
            .iter()
            .map(|item| item.uri.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != self.evidence.len()
        {
            return Err(MemoryRepositoryError::invariant(format!(
                "memory node {} contains duplicate evidence URIs",
                self.id
            )));
        }
        for relation in &self.relations {
            relation.validate(&self.id)?;
        }
        if self.relations.iter().collect::<BTreeSet<_>>().len() != self.relations.len() {
            return Err(MemoryRepositoryError::invariant(format!(
                "memory node {} contains duplicate relations",
                self.id
            )));
        }
        for (key, value) in &self.labels {
            validate_required_text("node.labels.key", key, MAX_LABEL_KEY_BYTES)?;
            validate_optional_text("node.labels.value", value, MAX_LABEL_VALUE_BYTES)?;
        }
        Ok(())
    }

    pub(crate) fn revision_updated_at(&self, revision: u64) -> Option<DateTime<Utc>> {
        if self.revision == revision {
            return Some(self.updated_at);
        }
        self.history
            .iter()
            .find(|item| item.revision == revision)
            .map(|item| item.updated_at)
    }
}

/// Caller-owned values used to create revision one of a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNodeDraft {
    pub id: String,
    pub namespace: MemoryNamespace,
    pub kind: DurableMemoryKind,
    pub status: MemoryStatus,
    pub content: String,
    pub confidence: f32,
    pub importance: f32,
    pub evidence: Vec<EvidenceRef>,
    pub relations: Vec<MemoryRelation>,
    pub labels: BTreeMap<String, String>,
    pub created_at: DateTime<Utc>,
}

impl MemoryNodeDraft {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        namespace: MemoryNamespace,
        kind: DurableMemoryKind,
        status: MemoryStatus,
        content: impl Into<String>,
        evidence: Vec<EvidenceRef>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: id.into(),
            namespace,
            kind,
            status,
            content: content.into(),
            confidence: 0.5,
            importance: 0.5,
            evidence,
            relations: Vec::new(),
            labels: BTreeMap::new(),
            created_at,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_relation(mut self, relation: MemoryRelation) -> Self {
        self.relations.push(relation);
        self
    }
}
