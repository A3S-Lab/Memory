//! DDL for the SQLite memory database.
//!
//! Tables:
//! - `memories`      — canonical store for all memory items
//! - `memories_fts`  — FTS5 virtual table (content table) for full-text search
//! - `session_log`   — append-only JSONL-equivalent event log per session
//!
//! Triggers keep `memories_fts` in sync with `memories` automatically.

/// Apply all DDL to an open connection.
pub fn apply(conn: &rusqlite::Connection) -> anyhow::Result<()> {
    conn.execute_batch(DDL)?;
    Ok(())
}

const DDL: &str = r#"
-- Enable WAL for better concurrent read/write performance.
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── memories ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT    PRIMARY KEY NOT NULL,
    content         TEXT    NOT NULL,
    timestamp_ms    INTEGER NOT NULL,   -- Unix ms (UTC)
    importance      REAL    NOT NULL DEFAULT 0.5,
    tags            TEXT    NOT NULL DEFAULT '[]',   -- JSON array
    memory_type     TEXT    NOT NULL DEFAULT 'episodic',
    metadata        TEXT    NOT NULL DEFAULT '{}',   -- JSON object
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed_ms INTEGER            -- NULL until first retrieval
);

CREATE INDEX IF NOT EXISTS idx_memories_timestamp  ON memories (timestamp_ms  DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance    DESC);
CREATE INDEX IF NOT EXISTS idx_memories_type       ON memories (memory_type);

-- ── FTS5 full-text index ────────────────────────────────────────────────────
-- content=memories keeps FTS in sync via triggers below.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 1'
);

-- Sync triggers ---
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts (rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts (rowid, content) VALUES (new.rowid, new.content);
END;

-- ── session_log ─────────────────────────────────────────────────────────────
-- Append-only JSONL-equivalent log; never modified after insert.
CREATE TABLE IF NOT EXISTS session_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    event_type   TEXT    NOT NULL,
    data_json    TEXT    NOT NULL DEFAULT '{}',
    timestamp_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_log_session ON session_log (session_id, timestamp_ms);
"#;
