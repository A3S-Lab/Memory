# a3s-memory

Pluggable memory storage for A3S.

Provides the `MemoryStore` trait and two default implementations. Agents that need to persist and recall knowledge across sessions depend on this crate directly — nothing else required.

The crate also provides a separate, dependency-free `VectorIndex` capability
for caller-owned ephemeral semantic retrieval. Vector indexes are not memory
stores: callers remain responsible for document admission, embedding
generation, lifecycle, and result fusion.

Default stores also enforce a small amount of memory hygiene: normalized exact
duplicates are merged into the existing item, tags/metadata/importance are
consolidated, and pruning protects curated memories such as pinned, frequently
recalled, consolidated, or conflict-tracking items. Semantic equivalence is not
inferred from keyword overlap at the storage layer.

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

### Ephemeral vector search

`InMemoryVectorIndex` stores caller-supplied vectors in immutable partition
snapshots and performs exact bounded top-k search. It does not use SQLite,
persist data, call an embedding model, or spawn a background task. The index is
released when its final owner is dropped.

```rust
use a3s_memory::{
    InMemoryVectorIndex, VectorIndex, VectorIndexDescriptor, VectorRecord,
    VectorSearchRequest,
};

let index = InMemoryVectorIndex::new(
    VectorIndexDescriptor::new(3)
        .with_max_records(10_000)
        .with_max_bytes(64 * 1024 * 1024),
)?;

index
    .replace_partition(
        "src/lib.rs",
        vec![VectorRecord::new("src/lib.rs:1-20", vec![0.8, 0.1, 0.2])
            .with_label("language", "rust")],
    )
    .await?;

let result = index
    .search(
        VectorSearchRequest::new(vec![0.7, 0.2, 0.1], 10)
            .with_label("language", "rust"),
    )
    .await?;
```

Dimensions are selected at index construction. Cosine indexes normalize
records and queries on admission, reject zero/non-finite vectors, and return
the immutable index revision that produced each result page. Replacing one
partition atomically publishes its complete new record set while sharing all
unchanged partition blocks.

Run the locked release qualification for 25,000 records at 384 dimensions with
`cargo run --example vector_search_benchmark --release`. It emits JSON evidence
and fails when exact top-20 search exceeds the 30 ms p95 budget.

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
durable duplicates after normalizing case and whitespace. Punctuation remains
significant. The first memory id remains canonical; later duplicates raise
importance, merge tags and list-style metadata such as `supersedes` /
`conflicts_with`, and record `duplicate_count` metadata.

Use `MemoryStore::store_and_return()` when the caller needs the canonical item
that now represents the fact. Semantic consolidation belongs to an upstream
model or caller with enough context to make that judgment. Such callers can use
`MemoryItem::merge_duplicate()` explicitly, or persist relation metadata and
let the owning memory runtime apply it.

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
| Workspace scanning, code chunking, embeddings, and hybrid ranking | `a3s-code` |

## Tests

The test suite covers `MemoryItem`, `RelevanceConfig`, `InMemoryStore`, and
`FileMemoryStore`, including persistence, index rebuild, path traversal
prevention, search specificity, exact duplicate consolidation, preservation of
distinct related memories, and protected pruning. Enabling the `sqlite` feature
also runs the SQLite backend contract.

```sh
cargo test
```

## Community

Join us on [Discord](https://discord.gg/XVg6Hu6H) for questions, discussions, and updates.

## License

MIT
