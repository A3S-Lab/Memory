//! Policy-free integrity kernel for durable agent memory.

mod access;
mod change;
mod change_engine;
mod change_token;
mod error;
mod file;
mod graph;
mod in_memory;
mod query;
mod snapshot;
mod store;
mod types;
mod validation;

pub use access::{MemoryAccessEvent, MemoryRepositorySnapshot, MemoryUsageSummary};
pub use change::{MemoryChangeResult, MemoryChangeSet, MemoryOperation};
pub use change_token::{MemoryNamespaceChangeToken, MEMORY_NAMESPACE_CHANGE_TOKEN_PROFILE_V1};
pub use error::MemoryRepositoryError;
pub use file::FileMemoryRepository;
pub use in_memory::InMemoryRepository;
pub use query::{
    MemoryQuery, MemoryQueryHit, MemoryQueryResult, MemoryScore, MEMORY_LEXICAL_QUERY_PROFILE_V1,
};
pub use snapshot::{
    MemoryNamespaceSnapshot, MemorySnapshotRequest, MAX_SNAPSHOT_BYTES, MAX_SNAPSHOT_NODES,
    MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1,
};
pub use store::MemoryRepository;
pub use types::{
    DurableMemoryKind, EvidenceKind, EvidenceRef, MemoryNamespace, MemoryNode, MemoryNodeDraft,
    MemoryNodeRevision, MemoryRelation, MemoryRelationKind, MemoryRevisionKind, MemoryStatus,
    RevisionMode,
};
pub use validation::{
    MAX_CHANGE_OPERATIONS, MAX_CONTENT_BYTES, MAX_EVIDENCE_PER_NODE, MAX_IDENTIFIER_BYTES,
    MAX_LABELS_PER_NODE, MAX_QUERY_LIMIT, MAX_RELATIONS_PER_NODE, MAX_REVISIONS_PER_NODE,
};
