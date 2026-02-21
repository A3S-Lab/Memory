# a3s-memory

Pluggable memory storage for A3S.

Provides the `MemoryStore` trait and two default implementations. Agents that need to persist and recall knowledge across sessions depend on this crate directly — nothing else required.

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

## What this crate does NOT own

| Concern | Lives in |
|---------|----------|
| Three-tier session memory (working / short-term / long-term) | `a3s-code` |
| `MemoryConfig` (max_short_term, max_working) | `a3s-code` |
| `MemoryStats` | `a3s-code` |
| Context injection into agent prompts | `a3s-code` |

## Tests

32 tests covering `MemoryItem`, `RelevanceConfig`, `InMemoryStore`, and `FileMemoryStore` (including persistence, index rebuild, and path traversal prevention).

```sh
cargo test
```

## Community

Join us on [Discord](https://discord.gg/XVg6Hu6H) for questions, discussions, and updates.

## License

MIT
