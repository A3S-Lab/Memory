//! A3S Memory — pluggable memory storage for AI agents.
//!
//! Provides the `MemoryStore` trait, `MemoryItem`, `MemoryType`,
//! configuration types, and a `FileMemoryStore` default implementation.

use anyhow::Context as _;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

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
        let mut items = self.items.write().await;
        if let Some(pos) = items.iter().position(|i| i.id == item.id) {
            items[pos] = item;
        } else {
            items.push(item);
        }
        Ok(())
    }

    async fn retrieve(&self, id: &str) -> anyhow::Result<Option<MemoryItem>> {
        Ok(self.items.read().await.iter().find(|i| i.id == id).cloned())
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let query_lower = query.to_lowercase();
        let config = RelevanceConfig::default();
        let now = Utc::now();
        let items = self.items.read().await;
        let mut matches: Vec<MemoryItem> = items
            .iter()
            .filter(|i| i.content_lower.contains(&query_lower))
            .cloned()
            .collect();
        matches.sort_by(|a, b| {
            b.relevance_score_at(now, &config)
                .partial_cmp(&a.relevance_score_at(now, &config))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(limit);
        Ok(matches)
    }

    async fn search_by_tags(
        &self,
        tags: &[String],
        limit: usize,
    ) -> anyhow::Result<Vec<MemoryItem>> {
        let config = RelevanceConfig::default();
        let now = Utc::now();
        let items = self.items.read().await;
        let mut matches: Vec<MemoryItem> = items
            .iter()
            .filter(|i| tags.iter().any(|t| i.tags.contains(t)))
            .cloned()
            .collect();
        matches.sort_by(|a, b| {
            b.relevance_score_at(now, &config)
                .partial_cmp(&a.relevance_score_at(now, &config))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(limit);
        Ok(matches)
    }

    async fn get_recent(&self, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let items = self.items.read().await;
        let mut sorted: Vec<MemoryItem> = items.iter().cloned().collect();
        sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
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

    async fn save_index(&self) -> anyhow::Result<()> {
        let index = self.index.read().await;
        let json = serde_json::to_string(&*index).context("Failed to serialize memory index")?;
        drop(index);
        let tmp = self.index_path.with_extension("json.tmp");
        tokio::fs::write(&tmp, json.as_bytes())
            .await
            .context("Failed to write memory index temp file")?;
        tokio::fs::rename(&tmp, &self.index_path)
            .await
            .context("Failed to rename memory index")?;
        Ok(())
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

    /// Rebuild the index from item files on disk (useful for corruption recovery).
    pub async fn rebuild_index(&self) -> anyhow::Result<usize> {
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
        *self.index.write().await = new_index;
        self.save_index().await?;
        Ok(count)
    }
}

#[async_trait::async_trait]
impl MemoryStore for FileMemoryStore {
    async fn store(&self, item: MemoryItem) -> anyhow::Result<()> {
        let mut item = item;
        item.id = Self::safe_id(&item.id);
        self.save_item(&item).await?;
        let entry = IndexEntry::from(&item);
        let mut index = self.index.write().await;
        if let Some(pos) = index.iter().position(|e| e.id == item.id) {
            index[pos] = entry;
        } else {
            index.push(entry);
        }
        drop(index);
        self.save_index().await
    }

    async fn retrieve(&self, id: &str) -> anyhow::Result<Option<MemoryItem>> {
        let path = self.item_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let data = tokio::fs::read_to_string(&path).await?;
        let mut item: MemoryItem = serde_json::from_str(&data)?;
        item.content_lower = item.content.to_lowercase();
        Ok(Some(item))
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let query_lower = query.to_lowercase();
        let index = self.index.read().await;
        let now = Utc::now();
        let config = RelevanceConfig::default();
        let mut matches: Vec<&IndexEntry> = index
            .iter()
            .filter(|e| e.content_lower.contains(&query_lower))
            .collect();
        matches.sort_by(|a, b| {
            index_score(a, now, &config)
                .partial_cmp(&index_score(b, now, &config))
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });
        let ids: Vec<String> = matches.iter().take(limit).map(|e| e.id.clone()).collect();
        drop(index);
        let mut items = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(item) = self.retrieve(&id).await? {
                items.push(item);
            }
        }
        sort_by_relevance(&mut items);
        Ok(items)
    }

    async fn search_by_tags(
        &self,
        tags: &[String],
        limit: usize,
    ) -> anyhow::Result<Vec<MemoryItem>> {
        let index = self.index.read().await;
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
        drop(index);
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
        let index = self.index.read().await;
        let mut sorted: Vec<&IndexEntry> = index.iter().collect();
        sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        let ids: Vec<String> = sorted.iter().take(limit).map(|e| e.id.clone()).collect();
        drop(index);
        let mut items = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(item) = self.retrieve(&id).await? {
                items.push(item);
            }
        }
        items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(items)
    }

    async fn get_important(&self, threshold: f32, limit: usize) -> anyhow::Result<Vec<MemoryItem>> {
        let index = self.index.read().await;
        let mut matches: Vec<&IndexEntry> =
            index.iter().filter(|e| e.importance >= threshold).collect();
        matches.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let ids: Vec<String> = matches.iter().take(limit).map(|e| e.id.clone()).collect();
        drop(index);
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
        let path = self.item_path(id);
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }
        let mut index = self.index.write().await;
        index.retain(|e| e.id != id);
        drop(index);
        self.save_index().await
    }

    async fn clear(&self) -> anyhow::Result<()> {
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
        Ok(self.index.read().await.len())
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
}
