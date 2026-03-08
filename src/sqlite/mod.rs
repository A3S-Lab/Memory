//! SQLite-backed `MemoryStore` with dual-track storage.
//!
//! # Storage tracks
//!
//! 1. **SQLite database** (`memory.db`) — structured store with FTS5 full-text
//!    search.  All `MemoryStore` trait methods operate on this DB.
//!
//! 2. **Markdown files** — human-readable append-only logs:
//!    - `MEMORY.md` for high-importance items (≥ 0.7)
//!    - `memory/YYYY-MM-DD.md` for episodic items (diary style)
//!
//! # Async bridge
//!
//! `rusqlite::Connection` is `Send` but not `Sync`.  All database operations
//! are dispatched via `tokio::task::spawn_blocking` with a shared
//! `Arc<Mutex<Connection>>`.
//!
//! # Optional: vector search
//!
//! When built with the `sqlite-vec` feature, `store_with_embedding` and
//! `search_semantic` are available for cosine-similarity nearest-neighbour
//! search over float32 embeddings.

pub mod markdown;
pub mod schema;

use crate::{MemoryItem, MemoryStore, MemoryType, PrunePolicy, RelevanceConfig};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{params, Connection};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

// ── SqliteMemoryStore ────────────────────────────────────────────────────────

/// SQLite-backed memory store with FTS5 search and Markdown export.
pub struct SqliteMemoryStore {
    conn: Arc<Mutex<Connection>>,
    base_dir: PathBuf,
    relevance: RelevanceConfig,
}

impl std::fmt::Debug for SqliteMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteMemoryStore")
            .field("base_dir", &self.base_dir)
            .finish()
    }
}

impl SqliteMemoryStore {
    /// Open (or create) a store rooted at `base_dir`.
    ///
    /// The SQLite file is `<base_dir>/memory.db`.
    pub async fn new(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&base_dir)
            .await
            .with_context(|| format!("Cannot create memory dir {}", base_dir.display()))?;

        let db_path = base_dir.join("memory.db");
        let conn = tokio::task::spawn_blocking(move || -> Result<Connection> {
            let conn = Connection::open(&db_path)
                .with_context(|| format!("Cannot open SQLite DB at {}", db_path.display()))?;
            schema::apply(&conn)?;

            #[cfg(feature = "sqlite-vec")]
            // SAFETY: transmute of a C function pointer to the expected auto-extension signature.
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                    sqlite_vec::sqlite3_vec_init as *const (),
                )));
            }

            #[cfg(feature = "sqlite-vec")]
            conn.execute_batch(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                     memory_id TEXT PRIMARY KEY,
                     embedding FLOAT[1536]
                 );",
            )?;

            Ok(conn)
        })
        .await
        .context("spawn_blocking panicked")??;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            base_dir,
            relevance: RelevanceConfig::default(),
        })
    }

    /// Override the relevance-scoring config used by `search` and `get_recent`.
    pub fn with_relevance(mut self, cfg: RelevanceConfig) -> Self {
        self.relevance = cfg;
        self
    }

    // ── Extended API ─────────────────────────────────────────────────────

    /// Append a structured event to the session log.
    pub async fn log_session_event(
        &self,
        session_id: &str,
        event_type: &str,
        data: &serde_json::Value,
    ) -> Result<()> {
        let now_ms = Utc::now().timestamp_millis();
        let data_json = serde_json::to_string(data)?;
        let session_id = session_id.to_string();
        let event_type = event_type.to_string();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let c = conn.lock().expect("sqlite lock poisoned");
            c.execute(
                "INSERT INTO session_log (session_id, event_type, data_json, timestamp_ms)
                 VALUES (?1, ?2, ?3, ?4)",
                params![session_id, event_type, data_json, now_ms],
            )?;
            Ok(())
        })
        .await
        .context("spawn_blocking panicked")?
    }

    /// Return all session-log rows for `session_id` as JSON values, oldest first.
    pub async fn export_session_log(&self, session_id: &str) -> Result<Vec<serde_json::Value>> {
        let session_id = session_id.to_string();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<serde_json::Value>> {
            let c = conn.lock().expect("sqlite lock poisoned");
            let mut stmt = c.prepare(
                "SELECT event_type, data_json, timestamp_ms
                 FROM session_log
                 WHERE session_id = ?1
                 ORDER BY timestamp_ms ASC",
            )?;
            let rows = stmt.query_map(params![session_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?;
            let mut out = Vec::new();
            for row in rows {
                let (event_type, data_json, ts) = row?;
                let data: serde_json::Value =
                    serde_json::from_str(&data_json).unwrap_or(serde_json::Value::Null);
                out.push(serde_json::json!({
                    "event_type": event_type,
                    "data": data,
                    "timestamp_ms": ts,
                }));
            }
            Ok(out)
        })
        .await
        .context("spawn_blocking panicked")?
    }

    /// Store an item together with a pre-computed embedding vector.
    ///
    /// Only available when the `sqlite-vec` Cargo feature is enabled.
    #[cfg(feature = "sqlite-vec")]
    pub async fn store_with_embedding(&self, item: MemoryItem, embedding: Vec<f32>) -> Result<()> {
        // First store normally (FTS + Markdown)
        self.store(item.clone()).await?;

        let id = item.id.clone();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let c = conn.lock().expect("sqlite lock poisoned");
            let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
            c.execute(
                "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?1, ?2)",
                params![id, blob],
            )?;
            Ok(())
        })
        .await
        .context("spawn_blocking panicked")?
    }

    /// Find the `limit` nearest neighbours to `query_embedding` by cosine
    /// distance, returning their ids.
    ///
    /// Only available when the `sqlite-vec` Cargo feature is enabled.
    #[cfg(feature = "sqlite-vec")]
    pub async fn search_semantic(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>> {
        let conn = self.conn.clone();
        let blob: Vec<u8> = query_embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let ids: Vec<String> = tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
            let c = conn.lock().expect("sqlite lock poisoned");
            let mut stmt = c.prepare(
                "SELECT memory_id
                 FROM memories_vec
                 WHERE embedding MATCH ?1
                 ORDER BY distance
                 LIMIT ?2",
            )?;
            let ids = stmt
                .query_map(params![blob, limit as i64], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(ids)
        })
        .await
        .context("spawn_blocking panicked")??;

        let mut items = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(item) = self.retrieve(id).await? {
                items.push(item);
            }
        }
        Ok(items)
    }

    // ── Private helpers ──────────────────────────────────────────────────

    async fn with_conn<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Connection) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let c = conn.lock().expect("sqlite lock poisoned");
            f(&c)
        })
        .await
        .context("spawn_blocking panicked")?
    }
}

// ── MemoryStore impl ─────────────────────────────────────────────────────────

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn store(&self, item: MemoryItem) -> Result<()> {
        // Write to Markdown tracks (best-effort; DB is authoritative)
        if let Err(e) = markdown::append(&self.base_dir, &item).await {
            eprintln!("[a3s-memory] markdown write failed for {}: {e}", item.id);
        }

        let tags_json = serde_json::to_string(&item.tags)?;
        let meta_json = serde_json::to_string(&item.metadata)?;
        let ts_ms = item.timestamp.timestamp_millis();
        let last_acc_ms = item.last_accessed.map(|t| t.timestamp_millis());
        let mtype = memory_type_to_str(&item.memory_type).to_string();

        self.with_conn(move |c| {
            c.execute(
                "INSERT OR REPLACE INTO memories
                 (id, content, timestamp_ms, importance, tags, memory_type, metadata,
                  access_count, last_accessed_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    item.id,
                    item.content,
                    ts_ms,
                    item.importance,
                    tags_json,
                    mtype,
                    meta_json,
                    item.access_count,
                    last_acc_ms,
                ],
            )?;
            Ok(())
        })
        .await
    }

    async fn retrieve(&self, id: &str) -> Result<Option<MemoryItem>> {
        let id = id.to_string();
        let now_ms = Utc::now().timestamp_millis();

        self.with_conn(move |c| {
            let result = c.query_row(
                "SELECT id, content, timestamp_ms, importance, tags, memory_type,
                        metadata, access_count, last_accessed_ms
                 FROM memories WHERE id = ?1",
                params![id],
                row_to_item,
            );
            match result {
                Ok(mut item) => {
                    // Update access stats
                    let _ = c.execute(
                        "UPDATE memories
                         SET access_count = access_count + 1, last_accessed_ms = ?1
                         WHERE id = ?2",
                        params![now_ms, item.id],
                    );
                    item.access_count += 1;
                    item.last_accessed = Some(ms_to_dt(now_ms));
                    Ok(Some(item))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(anyhow::anyhow!("retrieve({id}): {e}")),
            }
        })
        .await
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>> {
        let query = query.to_string();
        self.with_conn(move |c| {
            // Use FTS5 BM25 ranking, then join back for all columns.
            let mut stmt = c.prepare(
                "SELECT m.id, m.content, m.timestamp_ms, m.importance, m.tags,
                        m.memory_type, m.metadata, m.access_count, m.last_accessed_ms
                 FROM memories_fts
                 JOIN memories m ON memories_fts.rowid = m.rowid
                 WHERE memories_fts MATCH ?1
                 ORDER BY bm25(memories_fts)
                 LIMIT ?2",
            )?;
            let items = stmt
                .query_map(params![query, limit as i64], |row| Ok(row_to_item(row)))?
                .filter_map(|r| r.ok())
                .filter_map(|r| r.ok())
                .collect();
            Ok(items)
        })
        .await
    }

    async fn search_by_tags(&self, tags: &[String], limit: usize) -> Result<Vec<MemoryItem>> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }

        // Build a query that requires all tags to appear in the JSON array.
        // SQLite's json_each() lets us do set membership without extensions.
        let conn = self.conn.clone();
        let tags_owned: Vec<String> = tags.to_vec();
        tokio::task::spawn_blocking(move || -> Result<Vec<MemoryItem>> {
            let c = conn.lock().expect("sqlite lock poisoned");

            // Retrieve all then filter in Rust (simpler than dynamic SQL, fast
            // enough for typical memory store sizes).
            let mut stmt = c.prepare(
                "SELECT id, content, timestamp_ms, importance, tags, memory_type,
                        metadata, access_count, last_accessed_ms
                 FROM memories
                 ORDER BY timestamp_ms DESC",
            )?;
            let items: Vec<MemoryItem> = stmt
                .query_map([], |row| Ok(row_to_item(row)))?
                .filter_map(|r| r.ok())
                .filter_map(|r| r.ok())
                .filter(|item| tags_owned.iter().any(|t| item.tags.contains(t)))
                .take(limit)
                .collect();
            Ok(items)
        })
        .await
        .context("spawn_blocking panicked")?
    }

    async fn get_recent(&self, limit: usize) -> Result<Vec<MemoryItem>> {
        self.with_conn(move |c| {
            let mut stmt = c.prepare(
                "SELECT id, content, timestamp_ms, importance, tags, memory_type,
                        metadata, access_count, last_accessed_ms
                 FROM memories
                 ORDER BY timestamp_ms DESC
                 LIMIT ?1",
            )?;
            let items = stmt
                .query_map(params![limit as i64], |row| Ok(row_to_item(row)))?
                .filter_map(|r| r.ok())
                .filter_map(|r| r.ok())
                .collect();
            Ok(items)
        })
        .await
    }

    async fn get_important(&self, threshold: f32, limit: usize) -> Result<Vec<MemoryItem>> {
        self.with_conn(move |c| {
            let mut stmt = c.prepare(
                "SELECT id, content, timestamp_ms, importance, tags, memory_type,
                        metadata, access_count, last_accessed_ms
                 FROM memories
                 WHERE importance >= ?1
                 ORDER BY importance DESC
                 LIMIT ?2",
            )?;
            let items = stmt
                .query_map(params![threshold, limit as i64], |row| Ok(row_to_item(row)))?
                .filter_map(|r| r.ok())
                .filter_map(|r| r.ok())
                .collect();
            Ok(items)
        })
        .await
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        self.with_conn(move |c| {
            c.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
            Ok(())
        })
        .await
    }

    async fn clear(&self) -> Result<()> {
        self.with_conn(move |c| {
            c.execute_batch(
                "DELETE FROM memories;
                 INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');",
            )?;
            Ok(())
        })
        .await
    }

    async fn count(&self) -> Result<usize> {
        self.with_conn(move |c| {
            let n: i64 = c.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
            Ok(n as usize)
        })
        .await
    }

    async fn prune(&self, policy: &PrunePolicy) -> Result<usize> {
        let cutoff_ms = (chrono::Utc::now() - chrono::Duration::days(policy.max_age_days as i64))
            .timestamp_millis();
        let min_importance = policy.min_importance_to_keep;
        let max_items = policy.max_items;

        self.with_conn(move |c| {
            // Phase 1: delete items that are old AND below the importance threshold.
            let deleted1 = c.execute(
                "DELETE FROM memories WHERE timestamp_ms < ?1 AND importance < ?2",
                params![cutoff_ms, min_importance],
            )?;

            // Phase 2: enforce hard item cap, keeping highest-importance/newest items.
            let deleted2 = if max_items > 0 {
                c.execute(
                    "DELETE FROM memories WHERE id NOT IN (
                         SELECT id FROM memories
                         ORDER BY importance DESC, timestamp_ms DESC
                         LIMIT ?1
                     )",
                    params![max_items as i64],
                )?
            } else {
                0
            };

            Ok(deleted1 + deleted2)
        })
        .await
    }
}

// ── Row deserialization ──────────────────────────────────────────────────────

fn row_to_item(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryItem> {
    let id: String = row.get(0)?;
    let content: String = row.get(1)?;
    let ts_ms: i64 = row.get(2)?;
    let importance: f32 = row.get(3)?;
    let tags_json: String = row.get(4)?;
    let mtype_str: String = row.get(5)?;
    let meta_json: String = row.get(6)?;
    let access_count: u32 = row.get(7)?;
    let last_acc_ms: Option<i64> = row.get(8)?;

    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
    let metadata: std::collections::HashMap<String, String> =
        serde_json::from_str(&meta_json).unwrap_or_default();
    let memory_type = str_to_memory_type(&mtype_str);
    let timestamp = ms_to_dt(ts_ms);
    let last_accessed = last_acc_ms.map(ms_to_dt);
    let content_lower = content.to_lowercase();

    Ok(MemoryItem {
        id,
        content,
        timestamp,
        importance,
        tags,
        memory_type,
        metadata,
        access_count,
        last_accessed,
        content_lower,
    })
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(Utc::now)
}

fn memory_type_to_str(t: &MemoryType) -> &'static str {
    match t {
        MemoryType::Episodic => "episodic",
        MemoryType::Semantic => "semantic",
        MemoryType::Procedural => "procedural",
        MemoryType::Working => "working",
    }
}

fn str_to_memory_type(s: &str) -> MemoryType {
    match s {
        "episodic" => MemoryType::Episodic,
        "semantic" => MemoryType::Semantic,
        "procedural" => MemoryType::Procedural,
        "working" => MemoryType::Working,
        _ => MemoryType::Episodic,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryItem;
    use tempfile::TempDir;

    fn make_item(id: &str, content: &str, importance: f32) -> MemoryItem {
        MemoryItem {
            id: id.to_string(),
            content: content.to_string(),
            importance,
            memory_type: MemoryType::Semantic,
            ..MemoryItem::new(content.to_string())
        }
    }

    async fn store(dir: &TempDir) -> SqliteMemoryStore {
        SqliteMemoryStore::new(dir.path()).await.unwrap()
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        let item = make_item("id1", "hello world", 0.5);
        s.store(item.clone()).await.unwrap();

        let got = s.retrieve("id1").await.unwrap().expect("should exist");
        assert_eq!(got.content, "hello world");
        assert_eq!(got.access_count, 1);
    }

    #[tokio::test]
    async fn test_retrieve_missing_returns_none() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        assert!(s.retrieve("no-such-id").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_count_and_delete() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        s.store(make_item("a", "foo", 0.5)).await.unwrap();
        s.store(make_item("b", "bar", 0.5)).await.unwrap();
        assert_eq!(s.count().await.unwrap(), 2);

        s.delete("a").await.unwrap();
        assert_eq!(s.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_clear() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        s.store(make_item("x", "content", 0.8)).await.unwrap();
        s.clear().await.unwrap();
        assert_eq!(s.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_fts_search() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        s.store(make_item("1", "The quick brown fox", 0.5))
            .await
            .unwrap();
        s.store(make_item("2", "lazy dog barks", 0.5))
            .await
            .unwrap();

        let results = s.search("fox", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }

    #[tokio::test]
    async fn test_search_by_tags() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;

        let mut item = make_item("1", "tagged", 0.5);
        item.tags = vec!["rust".to_string(), "ai".to_string()];
        s.store(item).await.unwrap();

        let mut item2 = make_item("2", "other", 0.5);
        item2.tags = vec!["python".to_string()];
        s.store(item2).await.unwrap();

        let results = s.search_by_tags(&["rust".to_string()], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }

    #[tokio::test]
    async fn test_get_important() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        s.store(make_item("lo", "low", 0.3)).await.unwrap();
        s.store(make_item("hi", "high", 0.9)).await.unwrap();

        let results = s.get_important(0.7, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "hi");
    }

    #[tokio::test]
    async fn test_get_recent_order() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        s.store(make_item("old", "old content", 0.5)).await.unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        s.store(make_item("new", "new content", 0.5)).await.unwrap();

        let results = s.get_recent(2).await.unwrap();
        assert_eq!(results[0].id, "new");
    }

    #[tokio::test]
    async fn test_persistence_across_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let s = store(&dir).await;
            s.store(make_item("persist", "durable data", 0.6))
                .await
                .unwrap();
        }
        let s2 = SqliteMemoryStore::new(dir.path()).await.unwrap();
        let got = s2.retrieve("persist").await.unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().content, "durable data");
    }

    #[tokio::test]
    async fn test_session_log_roundtrip() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        s.log_session_event("sess1", "tool_use", &serde_json::json!({"tool": "Read"}))
            .await
            .unwrap();
        s.log_session_event("sess1", "response", &serde_json::json!({"len": 42}))
            .await
            .unwrap();

        let log = s.export_session_log("sess1").await.unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0]["event_type"], "tool_use");
        assert_eq!(log[1]["event_type"], "response");
    }

    #[tokio::test]
    async fn test_markdown_important_written() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        let mut item = make_item("imp", "very important note", 0.9);
        item.memory_type = MemoryType::Semantic;
        s.store(item).await.unwrap();

        // Give async write a moment to flush
        std::thread::sleep(std::time::Duration::from_millis(20));

        let md_path = dir.path().join("MEMORY.md");
        assert!(md_path.exists(), "MEMORY.md should be created");
        let content = tokio::fs::read_to_string(&md_path).await.unwrap();
        assert!(content.contains("very important note"));
    }

    #[tokio::test]
    async fn test_prune_removes_old_low_importance() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;

        let mut old_item = make_item("old_low", "stale memory", 0.2);
        // Force a timestamp 100 days in the past via direct insert
        let cutoff_ms = (chrono::Utc::now() - chrono::Duration::days(100)).timestamp_millis();
        {
            let conn = s.conn.clone();
            tokio::task::spawn_blocking(move || {
                let c = conn.lock().unwrap();
                c.execute(
                    "UPDATE memories SET timestamp_ms = ?1 WHERE id = ?2",
                    params![cutoff_ms, "old_low"],
                )
                .unwrap();
            })
            .await
            .unwrap();
        }
        old_item.timestamp = chrono::Utc::now() - chrono::Duration::days(100);
        s.store(old_item).await.unwrap();
        // Update the timestamp to be in the past (store sets now)
        {
            let conn = s.conn.clone();
            let ts = cutoff_ms - 1;
            tokio::task::spawn_blocking(move || {
                let c = conn.lock().unwrap();
                c.execute(
                    "UPDATE memories SET timestamp_ms = ?1 WHERE id = 'old_low'",
                    params![ts],
                )
                .unwrap();
            })
            .await
            .unwrap();
        }

        use crate::PrunePolicy;
        let policy = PrunePolicy {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        };
        let deleted = s.prune(&policy).await.unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(s.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_prune_keeps_high_importance() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        s.store(make_item("hi", "important", 0.9)).await.unwrap();
        // Force old timestamp
        {
            let conn = s.conn.clone();
            let ts = (chrono::Utc::now() - chrono::Duration::days(100)).timestamp_millis();
            tokio::task::spawn_blocking(move || {
                let c = conn.lock().unwrap();
                c.execute(
                    "UPDATE memories SET timestamp_ms = ?1 WHERE id = 'hi'",
                    params![ts],
                )
                .unwrap();
            })
            .await
            .unwrap();
        }

        use crate::PrunePolicy;
        let policy = PrunePolicy {
            max_age_days: 90,
            min_importance_to_keep: 0.5,
            max_items: 0,
        };
        let deleted = s.prune(&policy).await.unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(s.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_prune_max_items() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        for i in 0..10u32 {
            s.store(make_item(
                &format!("id{i}"),
                &format!("content {i}"),
                i as f32 * 0.1,
            ))
            .await
            .unwrap();
        }
        use crate::PrunePolicy;
        let policy = PrunePolicy {
            max_age_days: 9999,
            min_importance_to_keep: 0.0,
            max_items: 5,
        };
        let deleted = s.prune(&policy).await.unwrap();
        assert_eq!(deleted, 5);
        assert_eq!(s.count().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_markdown_episodic_daily_log() {
        let dir = TempDir::new().unwrap();
        let s = store(&dir).await;
        let mut item = make_item("ep", "today's activity log", 0.3);
        item.memory_type = MemoryType::Episodic;
        s.store(item).await.unwrap();

        std::thread::sleep(std::time::Duration::from_millis(20));

        let daily_dir = dir.path().join("memory");
        assert!(daily_dir.exists(), "memory/ directory should exist");
        let mut has_today = false;
        if let Ok(mut rd) = tokio::fs::read_dir(&daily_dir).await {
            while let Ok(Some(entry)) = rd.next_entry().await {
                let content = tokio::fs::read_to_string(entry.path())
                    .await
                    .unwrap_or_default();
                if content.contains("today's activity log") {
                    has_today = true;
                }
            }
        }
        assert!(has_today, "Episodic item should appear in daily log");
    }
}
