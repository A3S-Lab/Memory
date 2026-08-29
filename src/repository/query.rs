use super::validation::{
    validate_count, validate_required_text, MAX_CONTENT_BYTES, MAX_QUERY_LIMIT,
};
use super::{DurableMemoryKind, MemoryNamespace, MemoryNode, MemoryRepositoryError, MemoryStatus};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Exact-namespace repository query. Active status is the default view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryQuery {
    pub namespace: MemoryNamespace,
    pub text: Option<String>,
    pub kinds: BTreeSet<DurableMemoryKind>,
    pub statuses: BTreeSet<MemoryStatus>,
    pub limit: usize,
}

impl MemoryQuery {
    pub fn new(namespace: MemoryNamespace) -> Self {
        Self {
            namespace,
            text: None,
            kinds: BTreeSet::new(),
            statuses: BTreeSet::from([MemoryStatus::Active]),
            limit: 20,
        }
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_kinds(mut self, kinds: impl IntoIterator<Item = DurableMemoryKind>) -> Self {
        self.kinds = kinds.into_iter().collect();
        self
    }

    pub fn with_statuses(mut self, statuses: impl IntoIterator<Item = MemoryStatus>) -> Self {
        self.statuses = statuses.into_iter().collect();
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MemoryRepositoryError> {
        self.namespace.validate()?;
        if self.limit == 0 {
            return Err(MemoryRepositoryError::invalid(
                "query.limit",
                "must be greater than zero",
            ));
        }
        validate_count("query limit", self.limit, MAX_QUERY_LIMIT)?;
        if self.statuses.is_empty() {
            return Err(MemoryRepositoryError::invalid(
                "query.statuses",
                "must contain at least one status",
            ));
        }
        if let Some(text) = &self.text {
            validate_required_text("query.text", text, MAX_CONTENT_BYTES)?;
            if !text.chars().any(char::is_alphanumeric) {
                return Err(MemoryRepositoryError::invalid(
                    "query.text",
                    "must contain at least one alphanumeric term",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryScore {
    pub lexical: f32,
    pub total: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryQueryHit {
    pub node: MemoryNode,
    pub score: MemoryScore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryQueryResult {
    pub hits: Vec<MemoryQueryHit>,
}

pub(crate) fn query_nodes(
    nodes: Option<&BTreeMap<String, MemoryNode>>,
    query: &MemoryQuery,
) -> MemoryQueryResult {
    let query_tokens = query.text.as_deref().map(tokens);
    let mut hits = nodes
        .into_iter()
        .flat_map(BTreeMap::values)
        .filter(|node| query.statuses.contains(&node.status))
        .filter(|node| query.kinds.is_empty() || query.kinds.contains(&node.kind))
        .filter_map(|node| {
            let lexical = match &query_tokens {
                Some(terms) => lexical_score(terms, &node.content),
                None => 0.0,
            };
            if query_tokens.is_some() && lexical == 0.0 {
                return None;
            }
            Some(MemoryQueryHit {
                node: node.clone(),
                score: MemoryScore {
                    lexical,
                    total: lexical,
                },
            })
        })
        .collect::<Vec<_>>();

    hits.sort_by(|left, right| {
        right
            .score
            .total
            .total_cmp(&left.score.total)
            .then_with(|| right.node.updated_at.cmp(&left.node.updated_at))
            .then_with(|| left.node.id.cmp(&right.node.id))
    });
    hits.truncate(query.limit);
    MemoryQueryResult { hits }
}

fn lexical_score(query_tokens: &BTreeSet<String>, content: &str) -> f32 {
    let content_tokens = tokens(content);
    let matches = query_tokens.intersection(&content_tokens).count();
    matches as f32 / query_tokens.len() as f32
}

fn tokens(value: &str) -> BTreeSet<String> {
    value
        .to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}
