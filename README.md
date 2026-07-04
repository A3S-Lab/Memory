# a3s-memory

Pluggable memory storage for A3S.

Provides the `MemoryStore` trait and two default implementations. Agents that need to persist and recall knowledge across sessions depend on this crate directly — nothing else required.

Default stores also enforce a small amount of memory hygiene: exact and
conservative near-duplicate durable content is merged into the existing item,
tags/metadata/importance are consolidated, and pruning protects curated memories
such as pinned, frequently recalled, consolidated, or conflict-tracking items.

## Design

The crate follows a minimal core + external extensions pattern:

**Core (stable, non-replaceable):**
- `MemoryStore` — storage backend trait
- `MemoryItem` — the unit of memory
- `MemoryType` — episodic / semantic / procedural / working
- `RelevanceConfig` — scoring parameters

**Extensions (replaceable via `MemoryStore`):**
- `InMemoryStore` — default, ephemeral (testing and non-persistent use)
- `FileMemoryStore` — persistent, atomic writes, in-memory index

Three-tier session memory (`AgentMemory`) and context injection (`MemoryContextProvider`) live in `a3s-code`, not here. This crate only owns the storage layer.

## Usage

```toml
[dependencies]
a3s-memory = { version = "0.1", path = "../memory" }
```

### Store and retrieve

```rust
use a3s_memory::{InMemoryStore, MemoryItem, MemoryStore, MemoryType};
use std::sync::Arc;

let store = Arc::new(InMemoryStore::new());

let item = MemoryItem::new("Prefer write_all over write for file I/O")
    .with_importance(0.8)
    .with_tag("rust")
    .with_type(MemoryType::Semantic);

store.store(item).await?;

let results = store.search("file I/O", 5).await?;
```

### Persistent storage

```rust
use a3s_memory::{FileMemoryStore, MemoryStore};

let store = FileMemoryStore::new("/var/lib/agent/memory").await?;
// Directory layout:
//   memory/
//     index.json        ← in-memory index, persisted atomically
//     items/{id}.json   ← one file per memory item
```

### Custom backend

Implement `MemoryStore` to use any storage system (SQLite, vector DB, etc.):

```rust
use a3s_memory::{MemoryItem, MemoryStore};

struct MyStore { /* ... */ }

#[async_trait::async_trait]
impl MemoryStore for MyStore {
    async fn store(&self, item: MemoryItem) -> anyhow::Result<()> { todo!() }
    async fn retrieve(&self, id: &str) -> anyhow::Result<Option<MemoryItem>> { todo!() }
    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<MemoryItem>> { todo!() }
    // ... remaining methods
}
```

## Relevance scoring

Search combines lexical match strength (exact phrase, term, tag, and memory-type
matches) with the relevance score below. Exact or more specific query matches are
kept ahead of generic high-importance memories, while equally specific results
still benefit from importance and recency.

```
score = importance × importance_weight + decay × recency_weight
decay = exp(−age_days / decay_days)
```

Default: `importance_weight = 0.7`, `recency_weight = 0.3`, `decay_days = 30`.

```rust
use a3s_memory::{MemoryItem, RelevanceConfig};

let config = RelevanceConfig {
    decay_days: 7.0,        // faster decay
    importance_weight: 0.9,
    recency_weight: 0.1,
};

let score = item.relevance_score_at(now, &config);
```

## Deduplication and pruning

`InMemoryStore`, `FileMemoryStore`, and the optional SQLite store collapse exact
durable duplicates by a punctuation-insensitive content fingerprint. They also
merge conservative near-duplicates when the memory type matches, enough
non-stopword terms overlap, and the contents do not have conflicting negation
polarity. The first memory id remains canonical; later duplicates raise
importance, merge tags and list-style metadata such as `supersedes` /
`conflicts_with`, and record `duplicate_count` metadata.

Use `MemoryStore::store_and_return()` when the caller needs the canonical item
that now represents the fact. Callers that detect their own near-duplicates can
also use `MemoryItem::merge_duplicate()` and then store the returned canonical
item.

`PrunePolicy` removes old, low-importance items and can enforce a maximum item
count, but it hard-protects curated memories: `keep` / `pinned` / `protected`
tags or metadata, repeatedly accessed items, and memories carrying
`supersedes` / `conflicts_with` relation metadata.

## What this crate does NOT own

| Concern | Lives in |
|---------|----------|
| Three-tier session memory (working / short-term / long-term) | `a3s-code` |
| `MemoryConfig` (max_short_term, max_working) | `a3s-code` |
| `MemoryStats` | `a3s-code` |
| Context injection into agent prompts | `a3s-code` |

## Tests

89 default tests covering `MemoryItem`, `RelevanceConfig`, `InMemoryStore`, and
`FileMemoryStore` (including persistence, index rebuild, path traversal
prevention, search specificity, duplicate consolidation, conservative
near-duplicate merging, conflict-safe deduplication, and protected pruning).
With the `sqlite` feature enabled, the suite covers 108 tests including the
SQLite backend contract.

```sh
cargo test
```

## Community

Join us on [Discord](https://discord.gg/XVg6Hu6H) for questions, discussions, and updates.

## License

MIT
