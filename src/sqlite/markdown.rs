//! Dual-track Markdown writer.
//!
//! Two files are maintained alongside the SQLite database:
//!
//! - `MEMORY.md`                 — append-only log for high-importance items
//!                                  (importance ≥ `IMPORTANT_THRESHOLD`)
//! - `memory/YYYY-MM-DD.md`      — daily diary for `Episodic` memory items
//!
//! Both files are **append-only** — entries are never removed or rewritten.
//! The SQLite database is the authoritative source; these files are for
//! human-readable review and external tool consumption.

use crate::{MemoryItem, MemoryType};
use std::path::{Path, PathBuf};

/// Importance threshold for inclusion in `MEMORY.md`.
const IMPORTANT_THRESHOLD: f32 = 0.7;

/// Write `item` to the appropriate Markdown file(s) under `base_dir`.
///
/// Returns `Ok(())` even when nothing was written (item below all thresholds).
/// I/O errors are returned to the caller.
pub async fn append(base_dir: &Path, item: &MemoryItem) -> anyhow::Result<()> {
    if item.importance >= IMPORTANT_THRESHOLD {
        append_to_memory_md(base_dir, item).await?;
    }
    if item.memory_type == MemoryType::Episodic {
        append_to_daily_log(base_dir, item).await?;
    }
    Ok(())
}

async fn append_to_memory_md(base_dir: &Path, item: &MemoryItem) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let path = base_dir.join("MEMORY.md");
    let entry = format_entry(item);
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| anyhow::anyhow!("Cannot open {}: {e}", path.display()))?;
    f.write_all(format!("{entry}\n").as_bytes())
        .await
        .map_err(|e| anyhow::anyhow!("Write to {} failed: {e}", path.display()))?;
    Ok(())
}

async fn append_to_daily_log(base_dir: &Path, item: &MemoryItem) -> anyhow::Result<()> {
    let date = item.timestamp.format("%Y-%m-%d").to_string();
    let dir = base_dir.join("memory");
    tokio::fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("{date}.md"));

    use tokio::io::AsyncWriteExt;
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .map_err(|e| anyhow::anyhow!("Cannot open {}: {e}", path.display()))?;
    f.write_all(format!("{}\n", format_entry(item)).as_bytes())
        .await
        .map_err(|e| anyhow::anyhow!("Write to {} failed: {e}", path.display()))?;
    Ok(())
}

fn format_entry(item: &MemoryItem) -> String {
    let ts = item.timestamp.format("%Y-%m-%dT%H:%M:%SZ");
    let tags = if item.tags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", item.tags.join(", "))
    };
    format!(
        "## {ts} · {type_} · importance={importance:.2}{tags}\n\n{content}\n",
        ts = ts,
        type_ = format!("{:?}", item.memory_type).to_lowercase(),
        importance = item.importance,
        tags = tags,
        content = item.content.trim(),
    )
}

/// Return all Markdown file paths managed by this writer under `base_dir`.
/// Useful for testing / inspection.
pub fn list_files(base_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let main = base_dir.join("MEMORY.md");
    if main.exists() {
        out.push(main);
    }
    let daily_dir = base_dir.join("memory");
    if let Ok(rd) = std::fs::read_dir(&daily_dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().map_or(false, |e| e == "md") {
                out.push(p);
            }
        }
    }
    out
}
