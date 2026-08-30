use super::MemoryRepositoryError;
use serde::{Deserialize, Serialize};

/// Stable identity of the namespace change-token contract.
pub const MEMORY_NAMESPACE_CHANGE_TOKEN_PROFILE_V1: &str =
    "a3s.memory.namespace-change-token.sequence.v1";

/// Bounded, content-free evidence that one namespace did not change.
///
/// A token is meaningful only for repeated reads of the same exact namespace
/// from the same repository history. Equal tokens prove that no novel,
/// successful repository apply changed that namespace between the reads.
/// Sequences may jump, but must never repeat after a change. Admission and use
/// events do not change the token.
///
/// The token is not a namespace or backend identity, a snapshot, a lock, or a
/// distributed lease. Consumers must retain their ordinary snapshot and
/// revision-CAS proofs whenever the token is unavailable or changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNamespaceChangeToken {
    profile: String,
    sequence: u64,
}

impl MemoryNamespaceChangeToken {
    /// Construct a token for a backend-owned monotonic sequence.
    pub fn new(sequence: u64) -> Self {
        Self {
            profile: MEMORY_NAMESPACE_CHANGE_TOKEN_PROFILE_V1.to_string(),
            sequence,
        }
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Verify that a deserialized or backend-provided token uses this contract.
    pub fn verify(&self) -> Result<(), MemoryRepositoryError> {
        if self.profile != MEMORY_NAMESPACE_CHANGE_TOKEN_PROFILE_V1 {
            return Err(MemoryRepositoryError::invariant(
                "namespace change token uses an unsupported profile",
            ));
        }
        Ok(())
    }
}
