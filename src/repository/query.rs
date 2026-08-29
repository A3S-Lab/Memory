use super::validation::{
    validate_count, validate_required_text, MAX_CONTENT_BYTES, MAX_QUERY_LIMIT,
};
use super::{DurableMemoryKind, MemoryNamespace, MemoryNode, MemoryRepositoryError, MemoryStatus};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Stable identity of the deterministic V2 lexical query algorithm.
///
/// Hosts that persist serving policy can bind this value and reject silent
/// query-semantics drift after a binary upgrade.
pub const MEMORY_LEXICAL_QUERY_PROFILE_V1: &str = "a3s.memory.lexical.word-cjk-bigram.v1";

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
    let mut tokens = BTreeSet::new();
    let mut word = String::new();
    let mut cjk_run = Vec::new();

    for character in value.to_lowercase().chars() {
        if is_cjk_character(character) {
            flush_word(&mut tokens, &mut word);
            cjk_run.push(character);
        } else {
            flush_cjk_run(&mut tokens, &mut cjk_run);
            if character.is_alphanumeric() {
                word.push(character);
            } else {
                flush_word(&mut tokens, &mut word);
            }
        }
    }
    flush_word(&mut tokens, &mut word);
    flush_cjk_run(&mut tokens, &mut cjk_run);
    tokens
}

fn flush_word(tokens: &mut BTreeSet<String>, word: &mut String) {
    if !word.is_empty() {
        tokens.insert(std::mem::take(word));
    }
}

fn flush_cjk_run(tokens: &mut BTreeSet<String>, run: &mut Vec<char>) {
    if run.is_empty() {
        return;
    }
    tokens.insert(run.iter().collect());
    for pair in run.windows(2) {
        tokens.insert(pair.iter().collect());
    }
    run.clear();
}

fn is_cjk_character(character: char) -> bool {
    matches!(
        u32::from(character),
        0x1100..=0x11FF
            | 0x2E80..=0x2FFF
            | 0x3005..=0x3007
            | 0x3031..=0x3035
            | 0x3040..=0x30FF
            | 0x3100..=0x312F
            | 0x3130..=0x318F
            | 0x3190..=0x319F
            | 0x31A0..=0x31BF
            | 0x31F0..=0x31FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA960..=0xA97F
            | 0xAC00..=0xD7AF
            | 0xD7B0..=0xD7FF
            | 0xF900..=0xFAFF
            | 0xFF66..=0xFF9F
            | 0x20000..=0x2FA1F
            | 0x30000..=0x323AF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_preserves_words_and_adds_cjk_bigrams() {
        assert_eq!(
            tokens("Rust 数据库迁移 cargo_fmt"),
            BTreeSet::from([
                "rust".to_string(),
                "数据库迁移".to_string(),
                "数据".to_string(),
                "据库".to_string(),
                "库迁".to_string(),
                "迁移".to_string(),
                "cargo".to_string(),
                "fmt".to_string(),
            ])
        );
    }

    #[test]
    fn cjk_bigrams_cover_chinese_japanese_and_korean_without_unigrams() {
        for (input, expected) in [
            ("数据库", "数据"),
            ("確認手順", "確認"),
            ("배포절차", "배포"),
        ] {
            let actual = tokens(input);
            assert!(actual.contains(expected), "{input}: {actual:?}");
            assert!(!actual.contains(&input.chars().next().unwrap().to_string()));
        }
    }
}
