//! Caller-owned, ephemeral vector indexing primitives.
//!
//! The module deliberately does not generate embeddings or infer semantic
//! equivalence. A caller supplies admitted vectors and owns their lifecycle.

mod in_memory;
mod index;
mod search;
mod types;

#[cfg(test)]
mod tests;

pub use in_memory::InMemoryVectorIndex;
pub use index::VectorIndex;
pub use types::{
    VectorBudgetResource, VectorIndexChangeToken, VectorIndexDescriptor, VectorIndexError,
    VectorIndexStatus, VectorMetric, VectorMutationConsistency, VectorNormalization, VectorRecord,
    VectorResult, VectorRevision, VectorSearchHit, VectorSearchRequest, VectorSearchResult,
    VECTOR_INDEX_CHANGE_TOKEN_PROFILE_V1,
};
