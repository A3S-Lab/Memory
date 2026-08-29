use super::MemoryRepositoryError;

pub const MAX_IDENTIFIER_BYTES: usize = 256;
pub const MAX_CONTENT_BYTES: usize = 64 * 1024;
pub const MAX_EVIDENCE_PER_NODE: usize = 64;
pub const MAX_RELATIONS_PER_NODE: usize = 256;
pub const MAX_LABELS_PER_NODE: usize = 64;
pub const MAX_CHANGE_OPERATIONS: usize = 128;
pub const MAX_QUERY_LIMIT: usize = 100;
pub const MAX_REVISIONS_PER_NODE: usize = 1_024;

pub(crate) const MAX_URI_BYTES: usize = 2_048;
pub(crate) const MAX_DIGEST_BYTES: usize = 256;
pub(crate) const MAX_LABEL_KEY_BYTES: usize = 128;
pub(crate) const MAX_LABEL_VALUE_BYTES: usize = 1_024;

pub(crate) fn validate_required_text(
    field: &str,
    value: &str,
    max_bytes: usize,
) -> Result<(), MemoryRepositoryError> {
    if value.trim().is_empty() {
        return Err(MemoryRepositoryError::invalid(field, "must not be empty"));
    }
    validate_bytes(field, value.len(), max_bytes)
}

pub(crate) fn validate_optional_text(
    field: &str,
    value: &str,
    max_bytes: usize,
) -> Result<(), MemoryRepositoryError> {
    if value.is_empty() {
        return Ok(());
    }
    validate_required_text(field, value, max_bytes)
}

pub(crate) fn validate_count(
    resource: &str,
    actual: usize,
    limit: usize,
) -> Result<(), MemoryRepositoryError> {
    if actual > limit {
        return Err(MemoryRepositoryError::LimitExceeded {
            resource: resource.to_owned(),
            limit,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn validate_bytes(
    resource: &str,
    actual: usize,
    limit: usize,
) -> Result<(), MemoryRepositoryError> {
    validate_count(resource, actual, limit)
}

pub(crate) fn validate_unit_float(field: &str, value: f32) -> Result<(), MemoryRepositoryError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(MemoryRepositoryError::invalid(
            field,
            "must be a finite number between 0 and 1",
        ));
    }
    Ok(())
}
