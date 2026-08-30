//! Caller-owned vector indexing primitives.
//!
//! The module deliberately does not generate embeddings or infer semantic
//! equivalence. A caller supplies admitted vectors and owns their lifecycle.
//! The in-memory backend is dependency-free; the `sqlite` feature adds local
//! durability and cross-process revision compare-and-swap.

mod in_memory;
mod index;
mod search;
#[cfg(feature = "sqlite")]
mod sqlite;
mod types;

#[cfg(test)]
mod tests;

pub use in_memory::InMemoryVectorIndex;
pub use index::VectorIndex;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteVectorIndex;
pub use types::{
    VectorBudgetResource, VectorIndexChangeToken, VectorIndexDescriptor, VectorIndexError,
    VectorIndexObservation, VectorIndexStatus, VectorMetric, VectorMutationConsistency,
    VectorNormalization, VectorRecord, VectorResult, VectorRevision, VectorSearchHit,
    VectorSearchRequest, VectorSearchResult, VECTOR_INDEX_CHANGE_TOKEN_PROFILE_V1,
};
