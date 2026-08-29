use std::fmt;

/// Typed failures returned by the durable memory repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryRepositoryError {
    InvalidInput {
        field: String,
        message: String,
    },
    LimitExceeded {
        resource: String,
        limit: usize,
        actual: usize,
    },
    NamespaceMismatch {
        context: String,
    },
    NodeAlreadyExists {
        node_id: String,
    },
    NodeNotFound {
        node_id: String,
    },
    NodeRevisionNotFound {
        node_id: String,
        revision: u64,
    },
    RelationTargetNotFound {
        node_id: String,
        target_id: String,
    },
    EvidenceRequired {
        node_id: String,
    },
    AdmissionNotAllowed {
        node_id: String,
        revision: u64,
        current_revision: u64,
        current_status: String,
    },
    RevisionConflict {
        node_id: String,
        expected: u64,
        actual: u64,
    },
    IdempotencyConflict {
        key: String,
    },
    InvalidTransition {
        node_id: String,
        from: String,
        to: String,
    },
    InvariantViolation {
        message: String,
    },
    Persistence {
        operation: String,
        message: String,
    },
}

impl MemoryRepositoryError {
    pub(crate) fn invalid(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidInput {
            field: field.into(),
            message: message.into(),
        }
    }

    pub(crate) fn invariant(message: impl Into<String>) -> Self {
        Self::InvariantViolation {
            message: message.into(),
        }
    }
}

impl fmt::Display for MemoryRepositoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, message } => {
                write!(formatter, "invalid {field}: {message}")
            }
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => write!(
                formatter,
                "{resource} exceeds its limit of {limit} (actual: {actual})"
            ),
            Self::NamespaceMismatch { context } => {
                write!(formatter, "namespace mismatch in {context}")
            }
            Self::NodeAlreadyExists { node_id } => {
                write!(formatter, "memory node already exists: {node_id}")
            }
            Self::NodeNotFound { node_id } => write!(formatter, "memory node not found: {node_id}"),
            Self::NodeRevisionNotFound { node_id, revision } => write!(
                formatter,
                "memory node {node_id} revision was not found: {revision}"
            ),
            Self::RelationTargetNotFound { node_id, target_id } => write!(
                formatter,
                "relation target {target_id} for memory node {node_id} was not found"
            ),
            Self::EvidenceRequired { node_id } => {
                write!(formatter, "memory node {node_id} requires evidence")
            }
            Self::AdmissionNotAllowed {
                node_id,
                revision,
                current_revision,
                current_status,
            } => write!(
                formatter,
                "memory node {node_id} revision {revision} cannot be admitted; current revision {current_revision} is {current_status}"
            ),
            Self::RevisionConflict {
                node_id,
                expected,
                actual,
            } => write!(
                formatter,
                "memory node {node_id} revision conflict: expected {expected}, actual {actual}"
            ),
            Self::IdempotencyConflict { key } => {
                write!(
                    formatter,
                    "idempotency key was reused with different input: {key}"
                )
            }
            Self::InvalidTransition { node_id, from, to } => write!(
                formatter,
                "invalid status transition for memory node {node_id}: {from} -> {to}"
            ),
            Self::InvariantViolation { message } => {
                write!(formatter, "memory invariant violated: {message}")
            }
            Self::Persistence { operation, message } => {
                write!(
                    formatter,
                    "memory persistence {operation} failed: {message}"
                )
            }
        }
    }
}

impl std::error::Error for MemoryRepositoryError {}
