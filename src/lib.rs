//! A3S Memory — pluggable memory storage for AI agents.
//!
//! Provides the `MemoryStore` trait, `MemoryItem`, `MemoryType`,
//! configuration types, and the following backend implementations:
//!
//! | Backend | Feature | Search |
//! |---------|---------|--------|
//! | [`FileMemoryStore`] | always available | substring |
//! | [`SqliteMemoryStore`] | `sqlite` | BM25 via FTS5 |
//!
//! The [`FileMemoryStore`] is the default and requires no additional dependencies.

/// SQLite-backed memory store with dual-track Markdown export.
///
/// Requires the `sqlite` Cargo feature.
#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteMemoryStore;

use anyhow::Context as _;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use tokio::sync::RwLock;

const MIN_DEDUPE_FINGERPRINT_CHARS: usize = 24;
const MIN_NEAR_DEDUPE_TERMS: usize = 5;
const NEAR_DEDUPE_JACCARD_THRESHOLD: f32 = 0.86;
const PRUNE_PROTECTED_ACCESS_COUNT: u32 = 3;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelevanceConfig {
    /// Exponential decay half-life in days (default: 30.0)
    #[serde(default = "RelevanceConfig::default_decay_days")]
    pub decay_days: f32,
    /// Weight for importance factor (default: 0.7)
    #[serde(default = "RelevanceConfig::default_importance_weight")]
    pub importance_weight: f32,
    /// Weight for recency factor (default: 0.3)
    #[serde(default = "RelevanceConfig::default_recency_weight")]
    pub recency_weight: f32,
}

impl RelevanceConfig {
    fn default_decay_days() -> f32 {
        30.0
    }
    fn default_importance_weight() -> f32 {
        0.7
    }
    fn default_recency_weight() -> f32 {
        0.3
    }
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            decay_days: 30.0,
            importance_weight: 0.7,
            recency_weight: 0.3,
        }
    }
}

/// Policy controlling automatic pruning of long-term memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrunePolicy {
    /// Items older than this many days AND below `min_importance_to_keep` are deleted (default: 90).
    #[serde(default = "PrunePolicy::default_max_age_days")]
    pub max_age_days: u32,
    /// Items with importance below this threshold are eligible for age-based deletion (default: 0.5).
    /// High-importance items are never age-pruned.
    #[serde(default = "PrunePolicy::default_min_importance_to_keep")]
    pub min_importance_to_keep: f32,
    /// Hard cap on total items; when exceeded, lowest-relevance items are removed.
    /// 0 means unlimited (default: 0).
    #[serde(default)]
    pub max_items: usize,
}

impl PrunePolicy {
    fn default_max_age_days() -> u32 {
        90
    }
    fn default_min_importance_to_keep() -> f32 {
        0.5
    }
}

impl Default for PrunePolicy {
    fn default() -> Self {
        Self {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        }
    }
}

// ============================================================================
// Memory Item
// ============================================================================

/// A single memory item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub importance: f32,
    pub tags: Vec<String>,
    pub memory_type: MemoryType,
    pub metadata: HashMap<String, String>,
    pub access_count: u32,
    pub last_accessed: Option<DateTime<Utc>>,
    #[serde(skip)]
    pub content_lower: String,
}

impl MemoryItem {
    pub fn new(content: impl Into<String>) -> Self {
        let content = content.into();
        let content_lower = content.to_lowercase();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            timestamp: Utc::now(),
            importance: 0.5,
            tags: Vec::new(),
            memory_type: MemoryType::Episodic,
            metadata: HashMap::new(),
            access_count: 0,
            last_accessed: None,
            content_lower,
        }
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = memory_type;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Stable, punctuation-insensitive content fingerprint used by default
    /// stores to collapse exact durable duplicates.
    ///
    /// Very short memories return `None` so generic fragments such as "ok" or
    /// "done" are not accidentally merged.
    pub fn content_fingerprint(&self) -> Option<String> {
        memory_content_fingerprint(&self.content)
    }

    /// Merge a later duplicate observation into this canonical memory item.
    ///
    /// The canonical id is preserved while importance, tags, list-style
    /// metadata, access stats, and duplicate audit metadata are consolidated.
    pub fn merge_duplicate(self, incoming: MemoryItem) -> MemoryItem {
        merge_duplicate_memory_item(self, incoming)
    }

    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Some(Utc::now());
    }

    /// Calculate relevance score at a given timestamp using the provided config
    pub fn relevance_score_at(&self, now: DateTime<Utc>, config: &RelevanceConfig) -> f32 {
        let age_days = (now - self.timestamp).num_seconds() as f32 / 86400.0;
        let decay = (-age_days / config.decay_days).exp();
        self.importance * config.importance_weight + decay * config.recency_weight
    }

    /// Calculate relevance score with default config
    pub fn relevance_score(&self) -> f32 {
        self.relevance_score_at(Utc::now(), &RelevanceConfig::default())
    }
}

fn normalize_item_for_store(mut item: MemoryItem) -> MemoryItem {
    item.content_lower = item.content.to_lowercase();
    item
}

fn memory_content_fingerprint(content: &str) -> Option<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in content.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    let fingerprint = tokens.join(" ");
    if fingerprint.chars().count() < MIN_DEDUPE_FINGERPRINT_CHARS {
        None
    } else {
        Some(fingerprint)
    }
}

fn memories_are_store_duplicates(existing: &MemoryItem, incoming: &MemoryItem) -> bool {
    if existing.id == incoming.id {
        return true;
    }
    if existing.content_fingerprint().is_some()
        && existing.content_fingerprint() == incoming.content_fingerprint()
    {
        return true;
    }
    memory_items_are_near_duplicates(existing, incoming)
}

fn memory_index_entry_is_duplicate(entry: &IndexEntry, incoming: &MemoryItem) -> bool {
    if entry.id == incoming.id {
        return true;
    }
    if incoming.content_fingerprint().is_some()
        && memory_content_fingerprint(&entry.content_lower) == incoming.content_fingerprint()
    {
        return true;
    }
    memory_contents_are_near_duplicates(&entry.content_lower, entry.memory_type, incoming)
}

fn memory_items_are_near_duplicates(existing: &MemoryItem, incoming: &MemoryItem) -> bool {
    memory_contents_are_near_duplicates(&existing.content, existing.memory_type, incoming)
}

fn memory_contents_are_near_duplicates(
    existing_content: &str,
    existing_type: MemoryType,
    incoming: &MemoryItem,
) -> bool {
    if existing_type != incoming.memory_type {
        return false;
    }
    if has_conflicting_dedupe_polarity(existing_content, &incoming.content) {
        return false;
    }

    let existing_terms = dedupe_terms(existing_content);
    let incoming_terms = dedupe_terms(&incoming.content);
    if existing_terms.len() < MIN_NEAR_DEDUPE_TERMS || incoming_terms.len() < MIN_NEAR_DEDUPE_TERMS
    {
        return false;
    }

    let overlap = existing_terms.intersection(&incoming_terms).count();
    let union = existing_terms.len() + incoming_terms.len() - overlap;
    union > 0 && overlap as f32 / union as f32 >= NEAR_DEDUPE_JACCARD_THRESHOLD
}

fn dedupe_terms(content: &str) -> HashSet<String> {
    content
        .to_ascii_lowercase()
        .split(|ch: char| !(ch.is_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/')))
        .map(str::trim)
        .filter(|term| term.chars().count() >= 3)
        .filter(|term| !is_dedupe_stopword(term))
        .map(ToOwned::to_owned)
        .collect()
}

fn has_conflicting_dedupe_polarity(left: &str, right: &str) -> bool {
    has_negation_term(left) != has_negation_term(right)
}

fn has_negation_term(content: &str) -> bool {
    content
        .to_ascii_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .any(|term| {
            matches!(
                term,
                "not" | "never" | "no" | "avoid" | "without" | "disable"
            )
        })
}

fn is_dedupe_stopword(term: &str) -> bool {
    matches!(
        term,
        "the"
            | "and"
            | "for"
            | "with"
            | "after"
            | "before"
            | "from"
            | "that"
            | "this"
            | "when"
            | "then"
            | "than"
            | "into"
            | "must"
            | "should"
            | "would"
            | "could"
            | "about"
    )
}

fn merge_duplicate_memory_item(existing: MemoryItem, incoming: MemoryItem) -> MemoryItem {
    let incoming = normalize_item_for_store(incoming);
    let mut merged = existing.clone();

    if should_replace_duplicate_content(&existing, &incoming) {
        merged.content = incoming.content.clone();
        merged.content_lower = incoming.content_lower.clone();
    }
    merged.importance = existing.importance.max(incoming.importance);
    merged.timestamp = existing.timestamp.max(incoming.timestamp);
    merged.memory_type = stronger_memory_type(existing.memory_type, incoming.memory_type);
    merged.access_count = existing.access_count.max(incoming.access_count);
    merged.last_accessed = max_optional_datetime(existing.last_accessed, incoming.last_accessed);

    merge_tags(&mut merged.tags, &incoming.tags);
    merge_metadata(&mut merged.metadata, &incoming.metadata);
    record_duplicate_metadata(&mut merged.metadata, &incoming.id);

    normalize_item_for_store(merged)
}

fn should_replace_duplicate_content(existing: &MemoryItem, incoming: &MemoryItem) -> bool {
    incoming.importance > existing.importance
        || (incoming.importance == existing.importance
            && incoming.content.chars().count() > existing.content.chars().count())
}

fn stronger_memory_type(existing: MemoryType, incoming: MemoryType) -> MemoryType {
    if memory_type_strength(incoming) > memory_type_strength(existing) {
        incoming
    } else {
        existing
    }
}

fn memory_type_strength(memory_type: MemoryType) -> u8 {
    match memory_type {
        MemoryType::Procedural | MemoryType::Semantic => 3,
        MemoryType::Working => 2,
        MemoryType::Episodic => 1,
    }
}

fn max_optional_datetime(
    left: Option<DateTime<Utc>>,
    right: Option<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn merge_tags(existing: &mut Vec<String>, incoming: &[String]) {
    for tag in incoming {
        if !existing.contains(tag) {
            existing.push(tag.clone());
        }
    }
}

fn merge_metadata(existing: &mut HashMap<String, String>, incoming: &HashMap<String, String>) {
    for (key, value) in incoming {
        if value.trim().is_empty() {
            continue;
        }
        match existing.get_mut(key) {
            Some(current) if current == value => {}
            Some(current) if is_list_metadata_key(key) => {
                *current = merge_metadata_list(current, value);
            }
            Some(_) => {}
            None => {
                existing.insert(key.clone(), value.clone());
            }
        }
    }
}

fn is_list_metadata_key(key: &str) -> bool {
    matches!(
        key,
        "supersedes" | "conflicts_with" | "tools" | "aliases" | "entity_aliases"
    )
}

fn merge_metadata_list(existing: &str, incoming: &str) -> String {
    let mut values = Vec::new();
    for raw in existing.split(',').chain(incoming.split(',')) {
        let value = raw.trim();
        if !value.is_empty() && !values.iter().any(|seen| seen == value) {
            values.push(value.to_string());
        }
    }
    values.join(",")
}

fn record_duplicate_metadata(metadata: &mut HashMap<String, String>, incoming_id: &str) {
    let count = metadata
        .get("duplicate_count")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0)
        + 1;
    metadata.insert("duplicate_count".to_string(), count.to_string());
    metadata.insert("last_duplicate_at".to_string(), Utc::now().to_rfc3339());
    if !incoming_id.trim().is_empty() {
        let duplicate_ids = metadata
            .get("duplicate_ids")
            .map(|existing| merge_metadata_list(existing, incoming_id))
            .unwrap_or_else(|| incoming_id.to_string());
        metadata.insert("duplicate_ids".to_string(), duplicate_ids);
    }
}

fn memory_is_prune_protected(item: &MemoryItem) -> bool {
    item.access_count >= PRUNE_PROTECTED_ACCESS_COUNT
        || item.tags.iter().any(|tag| {
            matches!(
                tag.as_str(),
                "keep" | "pinned" | "protected" | "consolidated" | "conflict"
            )
        })
        || metadata_truthy(&item.metadata, "keep")
        || metadata_truthy(&item.metadata, "pinned")
        || metadata_truthy(&item.metadata, "protected")
        || metadata_nonempty(&item.metadata, "supersedes")
        || metadata_nonempty(&item.metadata, "conflicts_with")
}

fn metadata_truthy(metadata: &HashMap<String, String>, key: &str) -> bool {
    metadata
        .get(key)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "keep" | "pinned" | "protected"
            )
        })
        .unwrap_or(false)
}

fn metadata_nonempty(metadata: &HashMap<String, String>, key: &str) -> bool {
    metadata
        .get(key)
        .is_some_and(|value| !value.trim().is_empty())
}

/// Type of memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
    Working,
}

// ============================================================================
// Memory Store Trait
// ============================================================================

#[async_trait::async_trait]
pub trait MemoryStore: Send + Sync {
    async fn store(&self, item: MemoryItem) -> anyhow::Result<()>;
    /// Store a memory and return the canonical item that now represents it.
    ///
    /// Default stores may merge durable duplicate content into an existing item
    /// and return that existing item with updated metadata. Custom backends that
    /// do not implement deduplication can rely on this default wrapper.
    async fn store_and_return(&self, item: MemoryItem) -> anyhow::Result<MemoryItem> {
        self.store(item.clone()).await?;
        Ok(item)
    }
    async fn retrieve(&self, id: &str) -> anyhow::Result<Option<MemoryItem>>;
    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryItem>>;
    async fn search_by_tags(
        &self,
        tags: &[String],
        limit: usize,
    ) -> anyhow::Result<Vec<MemoryItem>>;
    async fn get_recent(&self, limit: usize) -> anyhow::Result<Vec<MemoryItem>>;
    async fn get_important(&self, threshold: f32, limit: usize) -> anyhow::Result<Vec<MemoryItem>>;
    async fn delete(&self, id: &str) -> anyhow::Result<()>;
    async fn clear(&self) -> anyhow::Result<()>;
    async fn count(&self) -> anyhow::Result<usize>;

    /// Remove stale or excess items according to `policy`.
    ///
    /// Returns the number of items deleted. The default implementation is a
    /// no-op (returns 0) for backwards compatibility.
    async fn prune(&self, policy: &PrunePolicy) -> anyhow::Result<usize> {
        let _ = policy;
        Ok(0)
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Score an index entry for sorting (avoids loading full MemoryItem from disk)
fn index_score(entry: &IndexEntry, now: DateTime<Utc>, config: &RelevanceConfig) -> f32 {
    let age_days = (now - entry.timestamp).num_seconds() as f32 / 86400.0;
    let decay = (-age_days / config.decay_days).exp();
    entry.importance * config.importance_weight + decay * config.recency_weight
}

fn sort_by_relevance(items: &mut [MemoryItem]) {
    let now = Utc::now();
    let config = RelevanceConfig::default();
    items.sort_by(|a, b| {
        b.relevance_score_at(now, &config)
            .partial_cmp(&a.relevance_score_at(now, &config))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn memory_type_to_query_key(memory_type: MemoryType) -> &'static str {
    match memory_type {
        MemoryType::Episodic => "episodic",
        MemoryType::Semantic => "semantic",
        MemoryType::Procedural => "procedural",
        MemoryType::Working => "working",
    }
}

fn query_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = query
        .to_lowercase()
        .split(|ch: char| {
            !(ch.is_alphanumeric() || matches!(ch, '/' | '\\' | '_' | '-' | '.' | ':' | '@'))
        })
        .map(str::trim)
        .filter(|term| term.chars().count() >= 2)
        .map(ToOwned::to_owned)
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

fn lexical_match_score(
    content_lower: &str,
    tags: &[String],
    memory_type: MemoryType,
    query_lower: &str,
    terms: &[String],
) -> Option<f32> {
    if query_lower.trim().is_empty() {
        return Some(0.0);
    }

    let mut score = 0.0;
    let mut matched_terms = 0usize;
    if !query_lower.is_empty() && content_lower.contains(query_lower) {
        score += 1.25;
    }

    let memory_type = memory_type_to_query_key(memory_type);
    for term in terms {
        let mut matched = false;
        if content_lower.contains(term) {
            score += 0.35;
            matched = true;
        }
        if tags.iter().any(|tag| tag.to_lowercase().contains(term)) {
            score += 0.55;
            matched = true;
        }
        if memory_type.contains(term) {
            score += 0.20;
            matched = true;
        }
        if matched {
            matched_terms += 1;
        }
    }

    if score <= 0.0 {
        return None;
    }

    if !terms.is_empty() {
        score += matched_terms as f32 / terms.len() as f32;
    }
    Some(score)
}

fn index_search_score(
    entry: &IndexEntry,
    now: DateTime<Utc>,
    config: &RelevanceConfig,
    query_lower: &str,
    terms: &[String],
) -> Option<f32> {
    let lexical = lexical_match_score(
        &entry.content_lower,
        &entry.tags,
        entry.memory_type,
        query_lower,
        terms,
    )?;
    Some(index_score(entry, now, config) + lexical)
}

// ============================================================================
// In-Memory Store
// ============================================================================

/// In-memory `MemoryStore` implementation.
///
/// Useful for testing and ephemeral (non-persistent) use cases.
pub struct InMemoryStore {
    items: RwLock<Vec<MemoryItem>>,
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            items: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait::async_trait]
impl MemoryStore for InMemoryStore {
    async fn store(&self, item: MemoryItem) -> anyhow::Result<()> {
        self.store_and_return(item).await.map(|_| ())
    }

    async fn store_and_return(&self, item: MemoryItem) -> anyhow::Result<MemoryItem> {
        let item = normalize_item_for_store(item);
        let mut items = self.items.write().await;
        if let Some(pos) = items.iter().position(|i| i.id == item.id) {
            items[pos] = item.clone();
            return Ok(item);
        }

        if let Some(pos) = items
            .iter()
            .position(|existing| memories_are_store_duplicates(existing, &item))
        {
            let merged = merge_duplicate_memory_item(items[pos].clone(), item);
            items[pos] = merged.clone();
            return Ok(merged);
        }

        items.push(item.clone());
        Ok(item)
    }

    async fn retrieve(&self, id: &str) -> anyhow::Result<Option<MemoryItem>> {
        let mut items = self.items.write().await;
        let Some(item) = items.iter_mut().find(|i| i.id == id) else {
            return Ok(None);
        };
        item.record_access();
        Ok(Some(item.clone()))
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let query_lower = query.to_lowercase();
        let config = RelevanceConfig::default();
        let now = Utc::now();
        let terms = query_terms(&query_lower);
        let mut items = self.items.write().await;
        let mut scored: Vec<(usize, f32)> = items
            .iter()
            .enumerate()
            .filter_map(|(idx, item)| {
                let lexical = lexical_match_score(
                    &item.content_lower,
                    &item.tags,
                    item.memory_type,
                    &query_lower,
                    &terms,
                )?;
                Some((idx, item.relevance_score_at(now, &config) + lexical))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| items[a.0].timestamp.cmp(&items[b.0].timestamp))
        });
        let ids: Vec<usize> = scored.into_iter().take(limit).map(|(idx, _)| idx).collect();
        let mut matches = Vec::with_capacity(ids.len());
        for idx in ids {
            items[idx].record_access();
            matches.push(items[idx].clone());
        }
        Ok(matches)
    }

    async fn search_by_tags(
        &self,
        tags: &[String],
        limit: usize,
    ) -> anyhow::Result<Vec<MemoryItem>> {
        let config = RelevanceConfig::default();
        let now = Utc::now();
        let mut items = self.items.write().await;
        let mut scored: Vec<(usize, f32)> = items
            .iter()
            .enumerate()
            .filter(|(_, item)| tags.iter().any(|t| item.tags.contains(t)))
            .map(|(idx, item)| (idx, item.relevance_score_at(now, &config)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| items[a.0].timestamp.cmp(&items[b.0].timestamp))
        });
        let ids: Vec<usize> = scored.into_iter().take(limit).map(|(idx, _)| idx).collect();
        let mut matches = Vec::with_capacity(ids.len());
        for idx in ids {
            items[idx].record_access();
            matches.push(items[idx].clone());
        }
        Ok(matches)
    }

    async fn get_recent(&self, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let items = self.items.read().await;
        let mut sorted: Vec<MemoryItem> = items.iter().cloned().collect();
        sorted.sort_by_key(|item| std::cmp::Reverse(item.timestamp));
        sorted.truncate(limit);
        Ok(sorted)
    }

    async fn get_important(&self, threshold: f32, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let items = self.items.read().await;
        let mut matches: Vec<MemoryItem> = items
            .iter()
            .filter(|i| i.importance >= threshold)
            .cloned()
            .collect();
        matches.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(limit);
        Ok(matches)
    }

    async fn delete(&self, id: &str) -> anyhow::Result<()> {
        self.items.write().await.retain(|i| i.id != id);
        Ok(())
    }

    async fn clear(&self) -> anyhow::Result<()> {
        self.items.write().await.clear();
        Ok(())
    }

    async fn count(&self) -> anyhow::Result<usize> {
        Ok(self.items.read().await.len())
    }

    async fn prune(&self, policy: &PrunePolicy) -> anyhow::Result<usize> {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::days(policy.max_age_days as i64);
        let min_importance = policy.min_importance_to_keep;

        let mut items = self.items.write().await;
        let before = items.len();

        // Phase 1: remove items that are old and below the importance threshold,
        // unless they have explicit curation/provenance protection.
        items.retain(|item| {
            memory_is_prune_protected(item)
                || item.importance >= min_importance
                || item.timestamp >= cutoff
        });

        // Phase 2: if still over the cap, keep protected memories first and then
        // the highest-relevance unprotected items.
        if policy.max_items > 0 && items.len() > policy.max_items {
            let config = RelevanceConfig::default();
            let protected_count = items
                .iter()
                .filter(|item| memory_is_prune_protected(item))
                .count();
            let unprotected_to_keep = policy.max_items.saturating_sub(protected_count);
            let mut unprotected_seen = 0usize;
            items.sort_by(|a, b| {
                memory_is_prune_protected(b)
                    .cmp(&memory_is_prune_protected(a))
                    .then_with(|| {
                        b.relevance_score_at(now, &config)
                            .partial_cmp(&a.relevance_score_at(now, &config))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            items.retain(|item| {
                if memory_is_prune_protected(item) {
                    true
                } else if unprotected_seen < unprotected_to_keep {
                    unprotected_seen += 1;
                    true
                } else {
                    false
                }
            });
        }

        Ok(before - items.len())
    }
}

// ============================================================================
// File-Based Memory Store
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    id: String,
    content_lower: String,
    tags: Vec<String>,
    importance: f32,
    timestamp: DateTime<Utc>,
    memory_type: MemoryType,
}

impl From<&MemoryItem> for IndexEntry {
    fn from(item: &MemoryItem) -> Self {
        Self {
            id: item.id.clone(),
            content_lower: item.content.to_lowercase(),
            tags: item.tags.clone(),
            importance: item.importance,
            timestamp: item.timestamp,
            memory_type: item.memory_type,
        }
    }
}

/// File-based memory store with atomic writes and in-memory index.
///
/// ```text
/// memory_dir/
///   index.json
///   items/{id}.json
/// ```
pub struct FileMemoryStore {
    items_dir: std::path::PathBuf,
    index_path: std::path::PathBuf,
    index: RwLock<Vec<IndexEntry>>,
}

static FILE_MEMORY_INDEX_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

fn file_memory_index_lock() -> &'static tokio::sync::Mutex<()> {
    FILE_MEMORY_INDEX_LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

impl FileMemoryStore {
    pub async fn new(dir: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let items_dir = dir.join("items");
        let index_path = dir.join("index.json");

        tokio::fs::create_dir_all(&items_dir)
            .await
            .with_context(|| {
                format!("Failed to create memory directory: {}", items_dir.display())
            })?;

        let index = if index_path.exists() {
            let data = tokio::fs::read_to_string(&index_path)
                .await
                .with_context(|| {
                    format!("Failed to read memory index: {}", index_path.display())
                })?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(Self {
            items_dir,
            index_path,
            index: RwLock::new(index),
        })
    }

    fn safe_id(id: &str) -> String {
        id.replace(['/', '\\'], "_").replace("..", "_")
    }

    fn item_path(&self, id: &str) -> std::path::PathBuf {
        self.items_dir.join(format!("{}.json", Self::safe_id(id)))
    }

    async fn read_index_from_disk(&self) -> anyhow::Result<Vec<IndexEntry>> {
        if !self.index_path.exists() {
            return Ok(Vec::new());
        }
        let data = tokio::fs::read_to_string(&self.index_path)
            .await
            .with_context(|| {
                format!("Failed to read memory index: {}", self.index_path.display())
            })?;
        Ok(serde_json::from_str(&data).unwrap_or_default())
    }

    async fn current_index(&self) -> Vec<IndexEntry> {
        match self.read_index_from_disk().await {
            Ok(index) => {
                *self.index.write().await = index.clone();
                index
            }
            Err(_) => self.index.read().await.clone(),
        }
    }

    async fn write_index_entries(&self, index: &[IndexEntry]) -> anyhow::Result<()> {
        let json = serde_json::to_string(index).context("Failed to serialize memory index")?;
        let tmp = self
            .index_path
            .with_extension(format!("json.{}.tmp", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, json.as_bytes())
            .await
            .context("Failed to write memory index temp file")?;
        tokio::fs::rename(&tmp, &self.index_path)
            .await
            .context("Failed to rename memory index")?;
        Ok(())
    }

    async fn save_index(&self) -> anyhow::Result<()> {
        let index = self.index.read().await.clone();
        self.write_index_entries(&index).await
    }

    async fn save_item(&self, item: &MemoryItem) -> anyhow::Result<()> {
        let path = self.item_path(&item.id);
        let json = serde_json::to_string_pretty(item)
            .with_context(|| format!("Failed to serialize memory item: {}", item.id))?;
        let tmp = path.with_extension("json.tmp");
        tokio::fs::write(&tmp, json.as_bytes())
            .await
            .with_context(|| format!("Failed to write memory item: {}", item.id))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .with_context(|| format!("Failed to rename memory item: {}", item.id))?;
        Ok(())
    }

    async fn load_item_without_access(&self, id: &str) -> anyhow::Result<Option<MemoryItem>> {
        let path = self.item_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let data = tokio::fs::read_to_string(&path).await?;
        let item: MemoryItem = serde_json::from_str(&data)?;
        Ok(Some(normalize_item_for_store(item)))
    }

    /// Rebuild the index from item files on disk (useful for corruption recovery).
    pub async fn rebuild_index(&self) -> anyhow::Result<usize> {
        let _guard = file_memory_index_lock().lock().await;
        let mut entries = tokio::fs::read_dir(&self.items_dir).await?;
        let mut new_index = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(data) = tokio::fs::read_to_string(&path).await {
                    if let Ok(item) = serde_json::from_str::<MemoryItem>(&data) {
                        new_index.push(IndexEntry::from(&item));
                    }
                }
            }
        }
        let count = new_index.len();
        self.write_index_entries(&new_index).await?;
        *self.index.write().await = new_index;
        Ok(count)
    }
}

#[async_trait::async_trait]
impl MemoryStore for FileMemoryStore {
    async fn store(&self, item: MemoryItem) -> anyhow::Result<()> {
        self.store_and_return(item).await.map(|_| ())
    }

    async fn store_and_return(&self, item: MemoryItem) -> anyhow::Result<MemoryItem> {
        let _guard = file_memory_index_lock().lock().await;
        let mut item = normalize_item_for_store(item);
        item.id = Self::safe_id(&item.id);
        let mut index = self.read_index_from_disk().await.unwrap_or_default();

        let duplicate_id = index
            .iter()
            .find(|entry| memory_index_entry_is_duplicate(entry, &item))
            .map(|entry| entry.id.clone());
        if let Some(duplicate_id) = duplicate_id {
            if duplicate_id != item.id {
                if let Some(existing) = self.load_item_without_access(&duplicate_id).await? {
                    if memories_are_store_duplicates(&existing, &item) {
                        let merged = merge_duplicate_memory_item(existing, item.clone());
                        self.save_item(&merged).await?;
                        if item.id != merged.id {
                            let stale_path = self.item_path(&item.id);
                            if stale_path.exists() {
                                let _ = tokio::fs::remove_file(stale_path).await;
                            }
                        }
                        index.retain(|entry| entry.id != item.id && entry.id != merged.id);
                        index.push(IndexEntry::from(&merged));
                        self.write_index_entries(&index).await?;
                        *self.index.write().await = index;
                        return Ok(merged);
                    }
                }
            }
        }

        self.save_item(&item).await?;
        let entry = IndexEntry::from(&item);
        if let Some(pos) = index.iter().position(|e| e.id == item.id) {
            index[pos] = entry;
        } else {
            index.push(entry);
        }
        self.write_index_entries(&index).await?;
        *self.index.write().await = index;
        Ok(item)
    }

    async fn retrieve(&self, id: &str) -> anyhow::Result<Option<MemoryItem>> {
        let path = self.item_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let data = tokio::fs::read_to_string(&path).await?;
        let mut item: MemoryItem = serde_json::from_str(&data)?;
        item.content_lower = item.content.to_lowercase();
        item.record_access();
        self.save_item(&item).await?;
        Ok(Some(item))
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let query_lower = query.to_lowercase();
        let index = self.current_index().await;
        let now = Utc::now();
        let config = RelevanceConfig::default();
        let terms = query_terms(&query_lower);
        let mut matches: Vec<(&IndexEntry, f32)> = index
            .iter()
            .filter_map(|e| {
                Some((
                    e,
                    index_search_score(e, now, &config, &query_lower, &terms)?,
                ))
            })
            .collect();
        matches.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.timestamp.cmp(&a.0.timestamp))
        });
        let ids: Vec<String> = matches
            .iter()
            .take(limit)
            .map(|(e, _)| e.id.clone())
            .collect();
        let mut items = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(item) = self.retrieve(&id).await? {
                items.push(item);
            }
        }
        Ok(items)
    }

    async fn search_by_tags(
        &self,
        tags: &[String],
        limit: usize,
    ) -> anyhow::Result<Vec<MemoryItem>> {
        let index = self.current_index().await;
        let now = Utc::now();
        let config = RelevanceConfig::default();
        let mut matches: Vec<&IndexEntry> = index
            .iter()
            .filter(|e| tags.iter().any(|t| e.tags.contains(t)))
            .collect();
        matches.sort_by(|a, b| {
            index_score(a, now, &config)
                .partial_cmp(&index_score(b, now, &config))
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });
        let ids: Vec<String> = matches.iter().take(limit).map(|e| e.id.clone()).collect();
        let mut items = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(item) = self.retrieve(&id).await? {
                items.push(item);
            }
        }
        sort_by_relevance(&mut items);
        Ok(items)
    }

    async fn get_recent(&self, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let index = self.current_index().await;
        let mut sorted: Vec<&IndexEntry> = index.iter().collect();
        sorted.sort_by_key(|entry| std::cmp::Reverse(entry.timestamp));
        let ids: Vec<String> = sorted.iter().take(limit).map(|e| e.id.clone()).collect();
        let mut items = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(item) = self.retrieve(&id).await? {
                items.push(item);
            }
        }
        items.sort_by_key(|item| std::cmp::Reverse(item.timestamp));
        Ok(items)
    }

    async fn get_important(&self, threshold: f32, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let index = self.current_index().await;
        let mut matches: Vec<&IndexEntry> =
            index.iter().filter(|e| e.importance >= threshold).collect();
        matches.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ids: Vec<String> = matches.iter().take(limit).map(|e| e.id.clone()).collect();
        let mut items = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(item) = self.retrieve(&id).await? {
                items.push(item);
            }
        }
        items.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(items)
    }

    async fn delete(&self, id: &str) -> anyhow::Result<()> {
        let _guard = file_memory_index_lock().lock().await;
        let path = self.item_path(id);
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }
        let mut index = self.read_index_from_disk().await.unwrap_or_default();
        index.retain(|e| e.id != id);
        self.write_index_entries(&index).await?;
        *self.index.write().await = index;
        Ok(())
    }

    async fn clear(&self) -> anyhow::Result<()> {
        let _guard = file_memory_index_lock().lock().await;
        let mut entries = tokio::fs::read_dir(&self.items_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let _ = tokio::fs::remove_file(&path).await;
            }
        }
        self.index.write().await.clear();
        self.save_index().await
    }

    async fn count(&self) -> anyhow::Result<usize> {
        Ok(self.current_index().await.len())
    }

    async fn prune(&self, policy: &PrunePolicy) -> anyhow::Result<usize> {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::days(policy.max_age_days as i64);
        let min_importance = policy.min_importance_to_keep;

        // Phase 1: collect IDs that are old and low-importance, unless the full
        // item carries curation/provenance protection.
        let mut items = Vec::new();
        for entry in self.current_index().await {
            if let Some(item) = self.load_item_without_access(&entry.id).await? {
                items.push(item);
            }
        }
        let phase1_ids: Vec<String> = items
            .iter()
            .filter(|item| {
                !memory_is_prune_protected(item)
                    && item.importance < min_importance
                    && item.timestamp < cutoff
            })
            .map(|item| item.id.clone())
            .collect();
        let mut deleted = phase1_ids.len();
        for id in &phase1_ids {
            self.delete(id).await?;
        }

        // Phase 2: enforce max_items cap by removing lowest-relevance
        // unprotected items. Protected items are hard-exempt, so the store may
        // remain above the cap if the user pinned more memories than the cap.
        if policy.max_items > 0 {
            let config = RelevanceConfig::default();
            let phase2_ids: Vec<String> = {
                let mut remaining = Vec::new();
                for entry in self.current_index().await {
                    if let Some(item) = self.load_item_without_access(&entry.id).await? {
                        remaining.push(item);
                    }
                }
                if remaining.len() <= policy.max_items {
                    Vec::new()
                } else {
                    let protected_count = remaining
                        .iter()
                        .filter(|item| memory_is_prune_protected(item))
                        .count();
                    let unprotected_to_keep = policy.max_items.saturating_sub(protected_count);
                    let mut unprotected: Vec<MemoryItem> = remaining
                        .into_iter()
                        .filter(|item| !memory_is_prune_protected(item))
                        .collect();
                    unprotected.sort_by(|a, b| {
                        b.relevance_score_at(now, &config)
                            .partial_cmp(&a.relevance_score_at(now, &config))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    unprotected
                        .into_iter()
                        .skip(unprotected_to_keep)
                        .map(|item| item.id)
                        .collect()
                }
            };
            deleted += phase2_ids.len();
            for id in &phase2_ids {
                self.delete(id).await?;
            }
        }

        Ok(deleted)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // MemoryItem tests

    #[test]
    fn test_memory_item_creation() {
        let item = MemoryItem::new("Test memory")
            .with_importance(0.8)
            .with_tag("test")
            .with_type(MemoryType::Semantic);
        assert_eq!(item.content, "Test memory");
        assert_eq!(item.importance, 0.8);
        assert_eq!(item.tags, vec!["test"]);
        assert_eq!(item.memory_type, MemoryType::Semantic);
    }

    #[test]
    fn test_memory_item_importance_clamped() {
        assert_eq!(MemoryItem::new("x").with_importance(1.5).importance, 1.0);
        assert_eq!(MemoryItem::new("x").with_importance(-0.5).importance, 0.0);
    }

    #[test]
    fn test_memory_item_record_access() {
        let mut item = MemoryItem::new("test");
        assert_eq!(item.access_count, 0);
        item.record_access();
        assert_eq!(item.access_count, 1);
        assert!(item.last_accessed.is_some());
    }

    #[test]
    fn test_memory_item_merge_duplicate_preserves_canonical_id() {
        let existing = MemoryItem::new("Run focused memory store tests after parser changes.")
            .with_importance(0.4)
            .with_tag("memory")
            .with_metadata("source", "workflow");
        let existing_id = existing.id.clone();
        let incoming =
            MemoryItem::new("Run focused memory store regression tests after parser changes.")
                .with_importance(0.9)
                .with_tag("tests")
                .with_metadata("supersedes", "old-memory");

        let merged = existing.merge_duplicate(incoming);

        assert_eq!(merged.id, existing_id);
        assert!(merged.content.contains("regression tests"));
        assert_eq!(merged.importance, 0.9);
        assert!(merged.tags.contains(&"memory".to_string()));
        assert!(merged.tags.contains(&"tests".to_string()));
        assert_eq!(
            merged.metadata.get("duplicate_count").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            merged.metadata.get("supersedes").map(String::as_str),
            Some("old-memory")
        );
    }

    #[test]
    fn test_memory_item_default_type_is_episodic() {
        assert_eq!(MemoryItem::new("test").memory_type, MemoryType::Episodic);
    }

    #[test]
    fn test_memory_item_all_types() {
        assert_eq!(
            MemoryItem::new("e")
                .with_type(MemoryType::Episodic)
                .memory_type,
            MemoryType::Episodic
        );
        assert_eq!(
            MemoryItem::new("s")
                .with_type(MemoryType::Semantic)
                .memory_type,
            MemoryType::Semantic
        );
        assert_eq!(
            MemoryItem::new("p")
                .with_type(MemoryType::Procedural)
                .memory_type,
            MemoryType::Procedural
        );
        assert_eq!(
            MemoryItem::new("w")
                .with_type(MemoryType::Working)
                .memory_type,
            MemoryType::Working
        );
    }

    // relevance_score_at tests

    #[test]
    fn test_relevance_score_uses_config() {
        let item = MemoryItem::new("test").with_importance(1.0);
        let now = Utc::now();

        // High importance weight → score dominated by importance
        let config_importance = RelevanceConfig {
            decay_days: 30.0,
            importance_weight: 0.9,
            recency_weight: 0.1,
        };
        let score = item.relevance_score_at(now, &config_importance);
        assert!(score > 0.95, "score was {score}");

        // Short decay → recent item still scores well
        let config_fast_decay = RelevanceConfig {
            decay_days: 1.0,
            importance_weight: 0.7,
            recency_weight: 0.3,
        };
        let score2 = item.relevance_score_at(now, &config_fast_decay);
        assert!(score2 > 0.9, "score was {score2}");
    }

    #[test]
    fn test_relevance_score_decays_with_age() {
        let mut old_item = MemoryItem::new("old").with_importance(0.5);
        old_item.timestamp = Utc::now() - chrono::Duration::days(60);
        let config = RelevanceConfig::default(); // 30-day half-life
        let score = old_item.relevance_score_at(Utc::now(), &config);
        // After 60 days (2 half-lives), decay ≈ exp(-2) ≈ 0.135
        // score ≈ 0.5*0.7 + 0.135*0.3 ≈ 0.39
        assert!(score < 0.45, "score was {score}");
    }

    #[test]
    fn test_relevance_score_default_uses_default_config() {
        let item = MemoryItem::new("test").with_importance(0.9);
        let score = item.relevance_score();
        assert!(score > 0.6);
    }

    // RelevanceConfig tests

    #[test]
    fn test_relevance_config_defaults() {
        let c = RelevanceConfig::default();
        assert_eq!(c.decay_days, 30.0);
        assert_eq!(c.importance_weight, 0.7);
        assert_eq!(c.recency_weight, 0.3);
    }

    // InMemoryStore tests

    #[tokio::test]
    async fn test_in_memory_store_retrieve() {
        let store = InMemoryStore::new();
        let item = MemoryItem::new("hello").with_tag("test");
        store.store(item.clone()).await.unwrap();
        let r = store.retrieve(&item.id).await.unwrap();
        assert!(r.is_some());
        assert_eq!(r.unwrap().content, "hello");
        let r = store.retrieve(&item.id).await.unwrap().unwrap();
        assert_eq!(r.access_count, 2);
    }

    #[tokio::test]
    async fn test_in_memory_store_retrieve_nonexistent() {
        let store = InMemoryStore::new();
        assert!(store.retrieve("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_in_memory_store_upsert() {
        let store = InMemoryStore::new();
        let mut item = MemoryItem::new("original");
        let id = item.id.clone();
        store.store(item.clone()).await.unwrap();
        item.content = "updated".to_string();
        item.content_lower = "updated".to_string();
        store.store(item).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
        assert_eq!(
            store.retrieve(&id).await.unwrap().unwrap().content,
            "updated"
        );
    }

    #[tokio::test]
    async fn test_in_memory_store_search_and_tags() {
        let store = InMemoryStore::new();
        store
            .store(MemoryItem::new("create file").with_tag("file"))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("delete file").with_tag("file"))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("create dir").with_tag("dir"))
            .await
            .unwrap();
        assert_eq!(store.search("create", 10).await.unwrap().len(), 2);
        assert_eq!(
            store
                .search_by_tags(&["file".to_string()], 10)
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn test_in_memory_store_search_relevance_order() {
        let store = InMemoryStore::new();
        store
            .store(MemoryItem::new("rust tip").with_importance(0.3))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("rust trick").with_importance(0.9))
            .await
            .unwrap();
        let results = store.search("rust", 10).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].importance >= results[1].importance);
    }

    #[tokio::test]
    async fn test_in_memory_store_delete_and_clear() {
        let store = InMemoryStore::new();
        let item = MemoryItem::new("to delete");
        let id = item.id.clone();
        store.store(item).await.unwrap();
        store.delete(&id).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);

        for i in 0..3 {
            store
                .store(MemoryItem::new(format!("item {i}")))
                .await
                .unwrap();
        }
        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_in_memory_store_get_recent() {
        let store = InMemoryStore::new();
        for i in 0..5 {
            let mut item = MemoryItem::new(format!("item {i}"));
            item.timestamp = Utc::now() + chrono::Duration::seconds(i as i64);
            store.store(item).await.unwrap();
        }
        let recent = store.get_recent(3).await.unwrap();
        assert_eq!(recent.len(), 3);
        assert!(recent[0].timestamp >= recent[1].timestamp);
    }

    #[tokio::test]
    async fn test_in_memory_store_get_important() {
        let store = InMemoryStore::new();
        store
            .store(MemoryItem::new("low").with_importance(0.2))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("high").with_importance(0.9))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("medium").with_importance(0.5))
            .await
            .unwrap();
        let results = store.get_important(0.7, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "high");
    }

    #[test]
    fn test_in_memory_store_default() {
        let _store: InMemoryStore = InMemoryStore::default();
    }

    // PrunePolicy / prune() tests

    #[test]
    fn test_prune_policy_defaults() {
        let p = PrunePolicy::default();
        assert_eq!(p.max_age_days, 90);
        assert_eq!(p.min_importance_to_keep, 0.5);
        assert_eq!(p.max_items, 0);
    }

    #[tokio::test]
    async fn test_prune_removes_old_low_importance() {
        let store = InMemoryStore::new();
        let mut old_item = MemoryItem::new("stale memory").with_importance(0.2);
        old_item.timestamp = Utc::now() - chrono::Duration::days(100);
        store.store(old_item).await.unwrap();

        let policy = PrunePolicy {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        };
        let deleted = store.prune(&policy).await.unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_prune_keeps_high_importance() {
        let store = InMemoryStore::new();
        let mut old_item = MemoryItem::new("important memory").with_importance(0.9);
        old_item.timestamp = Utc::now() - chrono::Duration::days(100);
        store.store(old_item).await.unwrap();

        let policy = PrunePolicy {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        };
        let deleted = store.prune(&policy).await.unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_prune_max_items() {
        let store = InMemoryStore::new();
        for i in 0..10 {
            store
                .store(MemoryItem::new(format!("item {i}")).with_importance(i as f32 * 0.1))
                .await
                .unwrap();
        }
        let policy = PrunePolicy {
            max_age_days: 9999,
            min_importance_to_keep: 0.0,
            max_items: 5,
        };
        let deleted = store.prune(&policy).await.unwrap();
        assert_eq!(deleted, 5);
        assert_eq!(store.count().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_prune_keeps_recent_low_importance() {
        // A recent item with low importance should NOT be pruned (not old enough)
        let store = InMemoryStore::new();
        store
            .store(MemoryItem::new("fresh").with_importance(0.1))
            .await
            .unwrap();

        let policy = PrunePolicy {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        };
        let deleted = store.prune(&policy).await.unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.count().await.unwrap(), 1);
    }
}

#[cfg(test)]
mod file_memory_store_tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (TempDir, FileMemoryStore) {
        let dir = TempDir::new().unwrap();
        let store = FileMemoryStore::new(dir.path()).await.unwrap();
        (dir, store)
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let (_dir, store) = setup().await;
        let item = MemoryItem::new("hello world");
        let id = item.id.clone();
        store.store(item).await.unwrap();
        let r = store.retrieve(&id).await.unwrap().unwrap();
        assert_eq!(r.content, "hello world");
    }

    #[tokio::test]
    async fn test_retrieve_nonexistent() {
        let (_dir, store) = setup().await;
        assert!(store.retrieve("nonexistent").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_search_by_content() {
        let (_dir, store) = setup().await;
        store
            .store(MemoryItem::new("rust programming"))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("python scripting"))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("rust async patterns"))
            .await
            .unwrap();
        let results = store.search("rust", 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_search_matches_non_contiguous_terms() {
        let (_dir, store) = setup().await;
        store
            .store(MemoryItem::new(
                "Success: release preflight\nTools: bash\nResult: provider verification passed",
            ))
            .await
            .unwrap();
        let results = store.search("release provider check", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("release preflight"));
    }

    #[tokio::test]
    async fn test_retrieve_records_access_on_disk() {
        let (dir, store) = setup().await;
        let item = MemoryItem::new("access me");
        let id = item.id.clone();
        store.store(item).await.unwrap();

        let item = store.retrieve(&id).await.unwrap().unwrap();
        assert_eq!(item.access_count, 1);

        let reopened = FileMemoryStore::new(dir.path()).await.unwrap();
        let item = reopened.retrieve(&id).await.unwrap().unwrap();
        assert_eq!(item.access_count, 2);
        assert!(item.last_accessed.is_some());
    }

    #[tokio::test]
    async fn test_search_limit() {
        let (_dir, store) = setup().await;
        for i in 0..10 {
            store
                .store(MemoryItem::new(format!("item {i}")))
                .await
                .unwrap();
        }
        assert_eq!(store.search("item", 3).await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_search_by_tags() {
        let (_dir, store) = setup().await;
        store
            .store(MemoryItem::new("one").with_tags(vec!["rust".into(), "async".into()]))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("two").with_tags(vec!["python".into()]))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("three").with_tags(vec!["rust".into()]))
            .await
            .unwrap();
        assert_eq!(
            store
                .search_by_tags(&["rust".to_string()], 10)
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn test_get_recent_ordered() {
        let (_dir, store) = setup().await;
        for i in 0..5 {
            let mut item = MemoryItem::new(format!("item {i}"));
            item.timestamp = Utc::now() + chrono::Duration::seconds(i as i64);
            store.store(item).await.unwrap();
        }
        let results = store.get_recent(3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].timestamp >= results[1].timestamp);
    }

    #[tokio::test]
    async fn test_get_important() {
        let (_dir, store) = setup().await;
        store
            .store(MemoryItem::new("low").with_importance(0.1))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("high").with_importance(0.9))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("medium").with_importance(0.5))
            .await
            .unwrap();
        let results = store.get_important(0.0, 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].importance >= results[1].importance);
    }

    #[tokio::test]
    async fn test_delete() {
        let (_dir, store) = setup().await;
        let item = MemoryItem::new("to delete");
        let id = item.id.clone();
        store.store(item).await.unwrap();
        store.delete(&id).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
        assert!(store.retrieve(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let (_dir, store) = setup().await;
        store.delete("nonexistent").await.unwrap();
    }

    #[tokio::test]
    async fn test_clear() {
        let (_dir, store) = setup().await;
        for i in 0..5 {
            store
                .store(MemoryItem::new(format!("item {i}")))
                .await
                .unwrap();
        }
        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_persistence_across_instances() {
        let dir = TempDir::new().unwrap();
        {
            let store = FileMemoryStore::new(dir.path()).await.unwrap();
            store
                .store(MemoryItem::new("persistent data").with_tags(vec!["test".into()]))
                .await
                .unwrap();
        }
        {
            let store = FileMemoryStore::new(dir.path()).await.unwrap();
            assert_eq!(store.count().await.unwrap(), 1);
            assert_eq!(store.search("persistent", 10).await.unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn test_stale_instances_merge_index_on_store() {
        let dir = TempDir::new().unwrap();
        let store_a = FileMemoryStore::new(dir.path()).await.unwrap();
        let store_b = FileMemoryStore::new(dir.path()).await.unwrap();

        store_a
            .store(MemoryItem::new("alpha stale merge"))
            .await
            .unwrap();
        store_b
            .store(MemoryItem::new("beta stale merge"))
            .await
            .unwrap();

        let reopened = FileMemoryStore::new(dir.path()).await.unwrap();
        assert_eq!(reopened.count().await.unwrap(), 2);
        assert_eq!(reopened.search("stale merge", 10).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_rebuild_index() {
        let dir = TempDir::new().unwrap();
        {
            let store = FileMemoryStore::new(dir.path()).await.unwrap();
            store.store(MemoryItem::new("alpha")).await.unwrap();
            store.store(MemoryItem::new("beta")).await.unwrap();
        }
        tokio::fs::remove_file(dir.path().join("index.json"))
            .await
            .unwrap();
        {
            let store = FileMemoryStore::new(dir.path()).await.unwrap();
            assert_eq!(store.count().await.unwrap(), 0);
            store.rebuild_index().await.unwrap();
            assert_eq!(store.count().await.unwrap(), 2);
        }
    }

    #[tokio::test]
    async fn test_path_traversal_prevention() {
        let (_dir, store) = setup().await;
        let mut item = MemoryItem::new("sneaky");
        item.id = "../../../etc/passwd".to_string();
        store.store(item).await.unwrap();
        let results = store.search("sneaky", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].id.contains('/'));
        assert!(!results[0].id.contains(".."));
    }

    #[tokio::test]
    async fn test_importance_threshold() {
        let (_dir, store) = setup().await;
        store
            .store(MemoryItem::new("low").with_importance(0.2))
            .await
            .unwrap();
        store
            .store(MemoryItem::new("high").with_importance(0.8))
            .await
            .unwrap();
        let results = store.get_important(0.5, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "high");
    }

    #[tokio::test]
    async fn test_file_prune_removes_old_low_importance() {
        let (_dir, store) = setup().await;
        let mut old_item = MemoryItem::new("stale").with_importance(0.2);
        old_item.timestamp = Utc::now() - chrono::Duration::days(100);
        store.store(old_item).await.unwrap();

        let policy = PrunePolicy {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        };
        let deleted = store.prune(&policy).await.unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_file_prune_keeps_high_importance() {
        let (_dir, store) = setup().await;
        let mut old_item = MemoryItem::new("important").with_importance(0.9);
        old_item.timestamp = Utc::now() - chrono::Duration::days(100);
        store.store(old_item).await.unwrap();

        let policy = PrunePolicy {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        };
        let deleted = store.prune(&policy).await.unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_file_prune_max_items() {
        let (_dir, store) = setup().await;
        for i in 0..10 {
            store
                .store(MemoryItem::new(format!("item {i}")).with_importance(i as f32 * 0.1))
                .await
                .unwrap();
        }
        let policy = PrunePolicy {
            max_age_days: 9999,
            min_importance_to_keep: 0.0,
            max_items: 5,
        };
        let deleted = store.prune(&policy).await.unwrap();
        assert_eq!(deleted, 5);
        assert_eq!(store.count().await.unwrap(), 5);
    }
}
