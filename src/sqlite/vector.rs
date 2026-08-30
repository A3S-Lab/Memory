use super::SqliteMemoryStore;
use crate::{MemoryItem, MemoryStore};
use anyhow::{Context, Result};
use rusqlite::params;

pub(super) fn register_auto_extension() -> Result<()> {
    static REGISTRATION: std::sync::OnceLock<std::result::Result<(), i32>> =
        std::sync::OnceLock::new();
    let registration = REGISTRATION.get_or_init(|| {
        // SAFETY: sqlite-vec exposes SQLite's extension initializer. SQLite
        // requires the equivalent no-argument auto-extension function pointer.
        #[allow(clippy::missing_transmute_annotations)]
        let result = unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )))
        };
        if result == rusqlite::ffi::SQLITE_OK {
            Ok(())
        } else {
            Err(result)
        }
    });
    registration.map_err(|code| {
        anyhow::anyhow!("Cannot register sqlite-vec auto extension (SQLite code {code})")
    })
}

impl SqliteMemoryStore {
    /// Store an item together with a pre-computed embedding vector.
    pub async fn store_with_embedding(&self, item: MemoryItem, embedding: Vec<f32>) -> Result<()> {
        let item = self.store_and_return(item).await?;

        let id = item.id.clone();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let c = conn.lock().expect("sqlite lock poisoned");
            let blob: Vec<u8> = embedding
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            c.execute(
                "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?1, ?2)",
                params![id, blob],
            )?;
            Ok(())
        })
        .await
        .context("spawn_blocking panicked")?
    }

    /// Find the nearest neighbours by cosine distance.
    pub async fn search_semantic(
        &self,
        query_embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>> {
        let conn = self.conn.clone();
        let blob: Vec<u8> = query_embedding
            .iter()
            .flat_map(|value| value.to_le_bytes())
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
                .filter_map(|result| result.ok())
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
}
