use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Similarity function used by an index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorMetric {
    /// Cosine similarity over unit-normalized vectors.
    Cosine,
    /// Raw dot product, or angular dot product when normalization is enabled.
    DotProduct,
}

/// Vector transformation applied at admission and query time.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorNormalization {
    /// Preserve caller-supplied finite values.
    None,
    /// Normalize every vector to unit L2 length.
    Unit,
}

/// Immutable vector shape and logical resource budgets.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorIndexDescriptor {
    pub dimension: usize,
    pub metric: VectorMetric,
    pub normalization: VectorNormalization,
    pub max_records: usize,
    pub max_bytes: usize,
}

impl VectorIndexDescriptor {
    pub const DEFAULT_MAX_RECORDS: usize = 100_000;
    pub const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

    /// Create a cosine index with caller-selected dimensions.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: VectorMetric::Cosine,
            normalization: VectorNormalization::Unit,
            max_records: Self::DEFAULT_MAX_RECORDS,
            max_bytes: Self::DEFAULT_MAX_BYTES,
        }
    }

    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_normalization(mut self, normalization: VectorNormalization) -> Self {
        self.normalization = normalization;
        self
    }

    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = max_records;
        self
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub(crate) fn validate(&self) -> VectorResult<()> {
        if self.dimension == 0 {
            return Err(VectorIndexError::InvalidDescriptor(
                "dimension must be greater than zero".to_string(),
            ));
        }
        if self.max_records == 0 {
            return Err(VectorIndexError::InvalidDescriptor(
                "max_records must be greater than zero".to_string(),
            ));
        }
        if self.max_bytes == 0 {
            return Err(VectorIndexError::InvalidDescriptor(
                "max_bytes must be greater than zero".to_string(),
            ));
        }
        if self.metric == VectorMetric::Cosine && self.normalization != VectorNormalization::Unit {
            return Err(VectorIndexError::InvalidDescriptor(
                "cosine indexes require unit normalization".to_string(),
            ));
        }
        let vector_bytes = self
            .dimension
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                VectorIndexError::InvalidDescriptor(
                    "dimension exceeds addressable vector storage".to_string(),
                )
            })?;
        if vector_bytes > self.max_bytes {
            return Err(VectorIndexError::InvalidDescriptor(
                "one vector exceeds max_bytes before metadata is accounted".to_string(),
            ));
        }
        Ok(())
    }
}

/// One caller-owned vector record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorRecord {
    pub id: String,
    pub embedding: Vec<f32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
}

impl VectorRecord {
    pub fn new(id: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            embedding,
            labels: BTreeMap::new(),
        }
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// Monotonic revision of one vector index.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct VectorRevision(u64);

impl VectorRevision {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }

    pub(crate) fn next(self) -> VectorResult<Self> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(VectorIndexError::RevisionExhausted)
    }
}

/// Stable profile for exact vector-index history tokens.
pub const VECTOR_INDEX_CHANGE_TOKEN_PROFILE_V1: &str = "a3s.memory.vector-index-change-token.v1";

/// Exact revision evidence scoped to one vector-index history.
///
/// A durable backend must retain the same history identity only while it can
/// guarantee that every revision still names the same linear mutation history.
/// Recreated, restored, or independently initialized storage must use a new
/// identity. Equality is therefore meaningful across process restarts only for
/// a backend that durably preserves this contract.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorIndexChangeToken {
    profile: String,
    history_digest: String,
    revision: VectorRevision,
}

impl VectorIndexChangeToken {
    /// Construct a token for a caller-owned index history.
    pub fn try_new(
        history_digest: impl Into<String>,
        revision: VectorRevision,
    ) -> VectorResult<Self> {
        let token = Self {
            profile: VECTOR_INDEX_CHANGE_TOKEN_PROFILE_V1.to_string(),
            history_digest: history_digest.into(),
            revision,
        };
        token.verify()?;
        Ok(token)
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn history_digest(&self) -> &str {
        &self.history_digest
    }

    pub fn revision(&self) -> VectorRevision {
        self.revision
    }

    /// Verify the bounded public representation after deserialization.
    pub fn verify(&self) -> VectorResult<()> {
        if self.profile != VECTOR_INDEX_CHANGE_TOKEN_PROFILE_V1 {
            return Err(VectorIndexError::InvalidChangeToken(
                "profile is unsupported".to_string(),
            ));
        }
        if !valid_sha256(&self.history_digest) {
            return Err(VectorIndexError::InvalidChangeToken(
                "history_digest must be canonical lowercase SHA-256".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn with_revision(&self, revision: VectorRevision) -> Self {
        Self {
            revision,
            ..self.clone()
        }
    }
}

fn valid_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

/// Strongest mutation ordering contract exposed by one vector backend.
///
/// `IndexRevisionCas` means the backend compares the expected global index
/// revision and publishes the partition mutation at the same linearization
/// point. Callers can therefore reject a delayed writer instead of replacing
/// a newer index generation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum VectorMutationConsistency {
    /// Partition replacement is atomic, but independently prepared writers are
    /// not ordered against one another.
    #[default]
    PartitionAtomic,
    /// Partition mutations can be conditioned on the global index revision.
    IndexRevisionCas,
}

/// Current logical size and published revision of an index.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorIndexStatus {
    pub revision: VectorRevision,
    pub partition_count: usize,
    pub record_count: usize,
    /// Accounted vector and caller metadata bytes in the current snapshot.
    pub byte_count: usize,
}

/// One self-consistent observation of a vector index revision.
///
/// A change token, when present, must identify the same revision as `status`.
/// Correctness-sensitive callers should prefer [`VectorIndex::observe`](super::VectorIndex::observe)
/// over composing the synchronous compatibility accessors themselves.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VectorIndexObservation {
    pub status: VectorIndexStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change_token: Option<VectorIndexChangeToken>,
}

impl VectorIndexObservation {
    /// Verify that all evidence names one valid index revision.
    pub fn verify(&self) -> VectorResult<()> {
        if let Some(token) = &self.change_token {
            token.verify()?;
            if token.revision() != self.status.revision {
                return Err(VectorIndexError::InvalidObservation(
                    "change token and status revisions differ".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// A bounded exact-vector query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSearchRequest {
    pub embedding: Vec<f32>,
    pub limit: usize,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub partitions: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
}

impl VectorSearchRequest {
    pub fn new(embedding: Vec<f32>, limit: usize) -> Self {
        Self {
            embedding,
            limit,
            partitions: BTreeSet::new(),
            labels: BTreeMap::new(),
        }
    }

    pub fn with_partition(mut self, partition: impl Into<String>) -> Self {
        self.partitions.insert(partition.into());
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// One ranked vector match.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSearchHit {
    pub id: String,
    pub partition: String,
    pub score: f32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub labels: BTreeMap<String, String>,
}

/// Ranked matches and the exact immutable revision that produced them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSearchResult {
    pub hits: Vec<VectorSearchHit>,
    pub status: VectorIndexStatus,
    pub searched_records: usize,
    pub truncated: bool,
}

/// Resource whose configured budget rejected a mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorBudgetResource {
    Records,
    Bytes,
}

/// Typed failures from vector admission, mutation, and search.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VectorIndexError {
    InvalidDescriptor(String),
    InvalidRequest(String),
    InvalidChangeToken(String),
    InvalidObservation(String),
    InvalidPartition,
    InvalidRecordId {
        partition: String,
        record_index: usize,
    },
    InvalidLabel {
        context: String,
    },
    DimensionMismatch {
        context: String,
        expected: usize,
        actual: usize,
    },
    NonFiniteVector {
        context: String,
        element_index: usize,
    },
    ZeroVector {
        context: String,
    },
    DuplicateRecordId {
        partition: String,
        id: String,
    },
    BudgetExceeded {
        resource: VectorBudgetResource,
        limit: usize,
        required: usize,
    },
    SizeOverflow,
    ScoreOverflow {
        partition: String,
        id: String,
    },
    ConditionalMutationUnsupported,
    RevisionConflict {
        expected: VectorRevision,
        actual: VectorRevision,
    },
    RevisionExhausted,
    DescriptorMismatch,
    StorageCorrupted(String),
    StorageFailed(String),
    WorkerFailed(String),
}

impl fmt::Display for VectorIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(message) => {
                write!(formatter, "invalid vector index: {message}")
            }
            Self::InvalidRequest(message) => write!(formatter, "invalid vector search: {message}"),
            Self::InvalidChangeToken(message) => {
                write!(formatter, "invalid vector index change token: {message}")
            }
            Self::InvalidObservation(message) => {
                write!(formatter, "invalid vector index observation: {message}")
            }
            Self::InvalidPartition => formatter.write_str("vector partition must not be empty"),
            Self::InvalidRecordId {
                partition,
                record_index,
            } => write!(
                formatter,
                "vector record {record_index} in partition '{partition}' has an empty id"
            ),
            Self::InvalidLabel { context } => {
                write!(formatter, "vector label key must not be empty ({context})")
            }
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "vector dimension mismatch for {context}: expected {expected}, got {actual}"
            ),
            Self::NonFiniteVector {
                context,
                element_index,
            } => write!(
                formatter,
                "vector for {context} contains a non-finite value at element {element_index}"
            ),
            Self::ZeroVector { context } => {
                write!(
                    formatter,
                    "vector for {context} cannot be normalized because it is zero"
                )
            }
            Self::DuplicateRecordId { partition, id } => write!(
                formatter,
                "partition '{partition}' contains duplicate vector record id '{id}'"
            ),
            Self::BudgetExceeded {
                resource,
                limit,
                required,
            } => write!(
                formatter,
                "vector {resource:?} budget exceeded: limit {limit}, required {required}"
            ),
            Self::SizeOverflow => formatter.write_str("vector index size accounting overflowed"),
            Self::ScoreOverflow { partition, id } => write!(
                formatter,
                "vector score overflowed for record '{id}' in partition '{partition}'"
            ),
            Self::ConditionalMutationUnsupported => {
                formatter.write_str("vector index does not support conditional partition mutation")
            }
            Self::RevisionConflict { expected, actual } => write!(
                formatter,
                "vector index revision conflict: expected {}, actual {}",
                expected.value(),
                actual.value()
            ),
            Self::RevisionExhausted => formatter.write_str("vector index revision exhausted"),
            Self::DescriptorMismatch => {
                formatter.write_str("vector index descriptor does not match durable storage")
            }
            Self::StorageCorrupted(message) => {
                write!(formatter, "vector index storage is corrupted: {message}")
            }
            Self::StorageFailed(message) => {
                write!(
                    formatter,
                    "vector index storage operation failed: {message}"
                )
            }
            Self::WorkerFailed(message) => write!(formatter, "vector worker failed: {message}"),
        }
    }
}

impl std::error::Error for VectorIndexError {}

pub type VectorResult<T> = Result<T, VectorIndexError>;
