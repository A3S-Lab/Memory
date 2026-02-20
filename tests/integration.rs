//! Integration tests for a3s-memory.
//!
//! These tests exercise the full public API across both storage backends,
//! verifying that `InMemoryStore` and `FileMemoryStore` behave identically
//! from a caller's perspective.

use a3s_memory::{
    FileMemoryStore, InMemoryStore, MemoryItem, MemoryStore, MemoryType, RelevanceConfig,
};
use std::sync::Arc;
use tempfile::TempDir;

// ============================================================================
// Shared contract tests — run against both backends
// ============================================================================

async fn contract_store_retrieve(store: &dyn MemoryStore) {
    let item = MemoryItem::new("integration test content")
        .with_importance(0.7)
        .with_tag("integration")
        .with_type(MemoryType::Semantic);
    let id = item.id.clone();

    store.store(item).await.unwrap();

    let retrieved = store.retrieve(&id).await.unwrap().unwrap();
    assert_eq!(retrieved.content, "integration test content");
    assert_eq!(retrieved.importance, 0.7);
    assert_eq!(retrieved.tags, vec!["integration"]);
    assert_eq!(retrieved.memory_type, MemoryType::Semantic);
}

async fn contract_retrieve_nonexistent(store: &dyn MemoryStore) {
    assert!(store.retrieve("does-not-exist").await.unwrap().is_none());
}

async fn contract_search(store: &dyn MemoryStore) {
    store
        .store(MemoryItem::new("rust async programming"))
        .await
        .unwrap();
    store
        .store(MemoryItem::new("rust error handling"))
        .await
        .unwrap();
    store
        .store(MemoryItem::new("python scripting"))
        .await
        .unwrap();

    let results = store.search("rust", 10).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.content.contains("rust")));
}

async fn contract_search_case_insensitive(store: &dyn MemoryStore) {
    store
        .store(MemoryItem::new("Rust Programming Language"))
        .await
        .unwrap();
    let results = store.search("rust", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

async fn contract_search_limit(store: &dyn MemoryStore) {
    for i in 0..10 {
        store
            .store(MemoryItem::new(format!("item {i}")))
            .await
            .unwrap();
    }
    let results = store.search("item", 3).await.unwrap();
    assert_eq!(results.len(), 3);
}

async fn contract_search_by_tags(store: &dyn MemoryStore) {
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

    let results = store
        .search_by_tags(&["rust".to_string()], 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);

    let results = store
        .search_by_tags(&["python".to_string()], 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);

    // No match
    let results = store.search_by_tags(&["go".to_string()], 10).await.unwrap();
    assert_eq!(results.len(), 0);
}

async fn contract_get_recent(store: &dyn MemoryStore) {
    use chrono::Utc;
    for i in 0..5 {
        let mut item = MemoryItem::new(format!("item {i}"));
        item.timestamp = Utc::now() + chrono::Duration::seconds(i as i64);
        store.store(item).await.unwrap();
    }
    let results = store.get_recent(3).await.unwrap();
    assert_eq!(results.len(), 3);
    assert!(results[0].timestamp >= results[1].timestamp);
    assert!(results[1].timestamp >= results[2].timestamp);
}

async fn contract_get_important(store: &dyn MemoryStore) {
    store
        .store(MemoryItem::new("low").with_importance(0.1))
        .await
        .unwrap();
    store
        .store(MemoryItem::new("medium").with_importance(0.5))
        .await
        .unwrap();
    store
        .store(MemoryItem::new("high").with_importance(0.9))
        .await
        .unwrap();

    let results = store.get_important(0.7, 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "high");

    let results = store.get_important(0.0, 2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].importance >= results[1].importance);
}

async fn contract_delete(store: &dyn MemoryStore) {
    let item = MemoryItem::new("to delete");
    let id = item.id.clone();
    store.store(item).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);

    store.delete(&id).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
    assert!(store.retrieve(&id).await.unwrap().is_none());
}

async fn contract_delete_nonexistent(store: &dyn MemoryStore) {
    // Must not error
    store.delete("nonexistent").await.unwrap();
}

async fn contract_clear(store: &dyn MemoryStore) {
    for i in 0..5 {
        store
            .store(MemoryItem::new(format!("item {i}")))
            .await
            .unwrap();
    }
    assert_eq!(store.count().await.unwrap(), 5);
    store.clear().await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}

async fn contract_count(store: &dyn MemoryStore) {
    assert_eq!(store.count().await.unwrap(), 0);
    store.store(MemoryItem::new("one")).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);
    store.store(MemoryItem::new("two")).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 2);
}

async fn contract_relevance_ordering(store: &dyn MemoryStore) {
    store
        .store(MemoryItem::new("rust tip").with_importance(0.2))
        .await
        .unwrap();
    store
        .store(MemoryItem::new("rust trick").with_importance(0.9))
        .await
        .unwrap();
    let results = store.search("rust", 10).await.unwrap();
    assert_eq!(results.len(), 2);
    // Higher importance should rank first (both items are equally recent)
    assert!(results[0].importance >= results[1].importance);
}

// ============================================================================
// InMemoryStore contract tests
// ============================================================================

macro_rules! in_memory_test {
    ($name:ident, $fn:ident) => {
        #[tokio::test]
        async fn $name() {
            $fn(&InMemoryStore::new()).await;
        }
    };
}

in_memory_test!(in_memory_store_retrieve, contract_store_retrieve);
in_memory_test!(
    in_memory_retrieve_nonexistent,
    contract_retrieve_nonexistent
);
in_memory_test!(in_memory_search, contract_search);
in_memory_test!(
    in_memory_search_case_insensitive,
    contract_search_case_insensitive
);
in_memory_test!(in_memory_search_limit, contract_search_limit);
in_memory_test!(in_memory_search_by_tags, contract_search_by_tags);
in_memory_test!(in_memory_get_recent, contract_get_recent);
in_memory_test!(in_memory_get_important, contract_get_important);
in_memory_test!(in_memory_delete, contract_delete);
in_memory_test!(in_memory_delete_nonexistent, contract_delete_nonexistent);
in_memory_test!(in_memory_clear, contract_clear);
in_memory_test!(in_memory_count, contract_count);
in_memory_test!(in_memory_relevance_ordering, contract_relevance_ordering);

// ============================================================================
// FileMemoryStore contract tests
// ============================================================================

macro_rules! file_store_test {
    ($name:ident, $fn:ident) => {
        #[tokio::test]
        async fn $name() {
            let dir = TempDir::new().unwrap();
            let store = FileMemoryStore::new(dir.path()).await.unwrap();
            $fn(&store).await;
        }
    };
}

file_store_test!(file_store_retrieve, contract_store_retrieve);
file_store_test!(file_retrieve_nonexistent, contract_retrieve_nonexistent);
file_store_test!(file_search, contract_search);
file_store_test!(
    file_search_case_insensitive,
    contract_search_case_insensitive
);
file_store_test!(file_search_limit, contract_search_limit);
file_store_test!(file_search_by_tags, contract_search_by_tags);
file_store_test!(file_get_recent, contract_get_recent);
file_store_test!(file_get_important, contract_get_important);
file_store_test!(file_delete, contract_delete);
file_store_test!(file_delete_nonexistent, contract_delete_nonexistent);
file_store_test!(file_clear, contract_clear);
file_store_test!(file_count, contract_count);
file_store_test!(file_relevance_ordering, contract_relevance_ordering);

// ============================================================================
// FileMemoryStore-specific tests
// ============================================================================

#[tokio::test]
async fn file_store_persists_across_instances() {
    let dir = TempDir::new().unwrap();
    {
        let store = FileMemoryStore::new(dir.path()).await.unwrap();
        store
            .store(MemoryItem::new("persistent").with_tags(vec!["test".into()]))
            .await
            .unwrap();
    }
    {
        let store = FileMemoryStore::new(dir.path()).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
        let results = store.search("persistent", 10).await.unwrap();
        assert_eq!(results[0].content, "persistent");
        assert_eq!(results[0].tags, vec!["test"]);
    }
}

#[tokio::test]
async fn file_store_rebuild_index_recovers_from_corruption() {
    let dir = TempDir::new().unwrap();
    {
        let store = FileMemoryStore::new(dir.path()).await.unwrap();
        store.store(MemoryItem::new("alpha")).await.unwrap();
        store.store(MemoryItem::new("beta")).await.unwrap();
    }
    // Simulate index corruption
    tokio::fs::remove_file(dir.path().join("index.json"))
        .await
        .unwrap();
    {
        let store = FileMemoryStore::new(dir.path()).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0); // index gone
        let recovered = store.rebuild_index().await.unwrap();
        assert_eq!(recovered, 2);
        assert_eq!(store.count().await.unwrap(), 2);
    }
}

#[tokio::test]
async fn file_store_path_traversal_prevention() {
    let dir = TempDir::new().unwrap();
    let store = FileMemoryStore::new(dir.path()).await.unwrap();
    let mut item = MemoryItem::new("sneaky");
    item.id = "../../../etc/passwd".to_string();
    store.store(item).await.unwrap();
    let results = store.search("sneaky", 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0].id.contains('/'));
    assert!(!results[0].id.contains(".."));
}

// ============================================================================
// RelevanceConfig tests
// ============================================================================

#[test]
fn relevance_config_custom_weights() {
    let item = MemoryItem::new("test").with_importance(1.0);
    let now = chrono::Utc::now();

    let aggressive = RelevanceConfig {
        decay_days: 1.0,
        importance_weight: 0.5,
        recency_weight: 0.5,
    };
    let conservative = RelevanceConfig {
        decay_days: 365.0,
        importance_weight: 0.9,
        recency_weight: 0.1,
    };

    let score_aggressive = item.relevance_score_at(now, &aggressive);
    let score_conservative = item.relevance_score_at(now, &conservative);

    // Both should be high for a brand-new item with max importance
    assert!(score_aggressive > 0.9);
    assert!(score_conservative > 0.9);
}

#[test]
fn relevance_score_old_item_decays() {
    let mut item = MemoryItem::new("old").with_importance(0.5);
    item.timestamp = chrono::Utc::now() - chrono::Duration::days(90);
    let config = RelevanceConfig::default(); // 30-day half-life
                                             // After 90 days (3 half-lives): decay ≈ exp(-3) ≈ 0.05
                                             // score ≈ 0.5*0.7 + 0.05*0.3 ≈ 0.365
    let score = item.relevance_score_at(chrono::Utc::now(), &config);
    assert!(score < 0.40, "score was {score}");
}

// ============================================================================
// MemoryItem builder API
// ============================================================================

#[test]
fn memory_item_builder_chain() {
    // with_tag appends; with_tags replaces
    let item = MemoryItem::new("content")
        .with_importance(0.8)
        .with_tags(vec!["a".into(), "b".into()])
        .with_tag("c")
        .with_type(MemoryType::Procedural)
        .with_metadata("key", "value");

    assert_eq!(item.content, "content");
    assert_eq!(item.importance, 0.8);
    assert_eq!(item.tags, vec!["a", "b", "c"]);
    assert_eq!(item.memory_type, MemoryType::Procedural);
    assert_eq!(item.metadata.get("key").unwrap(), "value");
}

#[test]
fn memory_item_with_tags_replaces() {
    let item = MemoryItem::new("x")
        .with_tag("a")
        .with_tag("b")
        .with_tags(vec!["c".into()]); // replaces previous tags
    assert_eq!(item.tags, vec!["c"]);
}

#[test]
fn memory_item_importance_clamped() {
    assert_eq!(MemoryItem::new("x").with_importance(2.0).importance, 1.0);
    assert_eq!(MemoryItem::new("x").with_importance(-1.0).importance, 0.0);
}

// ============================================================================
// Arc<dyn MemoryStore> usage (object safety check)
// ============================================================================

#[tokio::test]
async fn memory_store_is_object_safe() {
    let stores: Vec<Arc<dyn MemoryStore>> = vec![Arc::new(InMemoryStore::new())];
    for store in &stores {
        store.store(MemoryItem::new("test")).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
    }
}
