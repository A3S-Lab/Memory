# Smart Memory Retrieval — Development Plan

> Status: **PLANNED** — not yet implemented.
> Covers the work needed to add semantic vector search and automatic context injection to the a3s memory system.

---

## Background

Three retrieval mechanisms are needed for production-grade agent memory:

| # | Mechanism | Status |
|---|-----------|--------|
| 1 | **Keyword search** (FTS5 BM25) | ✅ Done — `SqliteMemoryStore::search()` |
| 2 | **Semantic / vector search** (embedding similarity) | 🟡 Storage layer done; embedding flow missing |
| 3 | **Smart context injection** (auto-inject before each LLM call) | ⬜ Not started |

This plan describes the remaining work for items 2 and 3.

---

## Architecture overview

```
User prompt
     │
     ▼
AgentLoop::run_turn()
     │
     ├──► resolve_context(prompt)           ← queries all ContextProviders in parallel
     │         │
     │         ▼
     │    SqliteMemoryContextProvider::query()
     │         │
     │         ├── FTS5 keyword search      ← SqliteMemoryStore::search()
     │         └── vec0 vector search       ← SqliteMemoryStore::search_semantic()
     │                   │
     │                   ▼
     │             fuse + rank results
     │
     ▼
build_augmented_system_prompt()
     │
     └──► injects top-k memories as <context type="memory" ...> XML blocks
                                 into the system prompt sent to the LLM
```

**Key types already in play:**

- `a3s-memory::SqliteMemoryStore` — storage (FTS5 + vec0)
- `a3s-code::context::ContextProvider` — injection trait (query + on_turn_complete)
- `a3s-code::context::EmbeddingProvider` — embedding model abstraction
- `a3s-code::context::VectorContextProvider` — generic vector RAG provider (reference)
- `a3s-code::memory::AgentMemory` — three-tier memory (working / short-term / long-term)
- `a3s-code::memory::MemoryContextProvider` — existing bridge (uses `AgentMemory`, not SQLite FTS5)

---

## Phase 1 — Wire SqliteMemoryStore into SafeClaw

**Goal:** Make SafeClaw use `SqliteMemoryStore` as the backing store for `AgentMemory`.
This alone gives FTS5 keyword search and Markdown export with no other changes.

### 1.1 Update `bootstrap.rs`

Replace `FileMemoryStore` (or `InMemoryStore`) initialisation in `init_memory_store()` with `SqliteMemoryStore`:

```rust
// crates/safeclaw/src/bootstrap.rs
use a3s_memory::SqliteMemoryStore;

pub async fn init_memory_store() -> Result<Arc<dyn MemoryStore>> {
    let dir = dirs_next::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("safeclaw")
        .join("memory");
    let store = SqliteMemoryStore::new(&dir).await?;
    Ok(Arc::new(store))
}
```

**Dependency:** Add `a3s-memory` with `sqlite` feature to `crates/safeclaw/Cargo.toml`.

### 1.2 Verify existing MemoryContextProvider picks it up

`AgentMemory` in `a3s-code` wraps any `Arc<dyn MemoryStore>`. Once the store is swapped, `MemoryContextProvider::query()` calls `store.search()` which now goes through FTS5 — no other changes needed.

**Tests to add:**
- Integration test: store a memory, trigger a new turn with a related prompt, assert the memory appears in the injected context XML.

---

## Phase 2 — Embedding pipeline (semantic search)

**Goal:** Compute and persist embedding vectors when a memory is stored; use them for cosine-similarity retrieval alongside FTS5.

### 2.1 `EmbeddingProvider` implementation

`a3s-code::context::EmbeddingProvider` is already defined. We need a concrete implementation backed by the configured LLM's embedding endpoint.

**Option A — LiteLLM-compatible HTTP client** (recommended)
Any OpenAI-compatible `/v1/embeddings` endpoint. Reuse the existing `OpenAiEmbeddingProvider` in `a3s-code`.

**Option B — Local model via `candle` or `ort`**
For fully offline operation. Higher complexity; defer unless privacy policy requires it.

### 2.2 `EmbeddingMemoryStore` wrapper

Wrap `SqliteMemoryStore` to intercept `store()` calls, compute an embedding, and persist it:

```rust
// proposed: crates/memory/src/sqlite/embedding_store.rs
pub struct EmbeddingMemoryStore<E: EmbeddingProvider> {
    inner: SqliteMemoryStore,
    embedder: Arc<E>,
}

#[async_trait]
impl MemoryStore for EmbeddingMemoryStore<E> {
    async fn store(&self, item: MemoryItem) -> Result<()> {
        let embedding = self.embedder.embed(&item.content).await?;
        self.inner.store_with_embedding(item, embedding.values).await
    }
    // delegate all other methods to self.inner
}
```

This keeps the embedding concern out of `SqliteMemoryStore` itself (separation of concerns).

**Note:** `EmbeddingProvider` is defined in `a3s-code`, so this wrapper either:
- Lives in `a3s-code` (colocated with the provider), or
- Uses a local trait mirror in `a3s-memory` with a blanket `impl` for `a3s-code`'s trait.

Decision: put the wrapper in `a3s-code/core/src/memory/embedding_store.rs` to avoid the cross-crate trait dependency.

### 2.3 `SqliteMemoryContextProvider`

Replace (or extend) the existing `MemoryContextProvider` with a version that performs hybrid search:

```rust
// proposed: a3s-code/core/src/memory/sqlite_context_provider.rs
pub struct SqliteMemoryContextProvider {
    store: Arc<SqliteMemoryStore>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    fts_weight: f32,   // default 0.4
    vec_weight: f32,   // default 0.6
    top_k: usize,      // default 5
}

impl ContextProvider for SqliteMemoryContextProvider {
    async fn query(&self, q: &ContextQuery) -> Result<ContextResult> {
        // 1. FTS5 keyword search
        let fts_hits = self.store.search(&q.query, self.top_k * 2).await?;

        // 2. Vector search (if embedder configured)
        let vec_hits = if let Some(ref e) = self.embedder {
            let emb = e.embed(&q.query).await?;
            self.store.search_semantic(emb.values, self.top_k * 2).await?
        } else {
            vec![]
        };

        // 3. Fuse: reciprocal rank fusion (RRF) or weighted score merge
        let fused = rrf_fuse(fts_hits, vec_hits, self.fts_weight, self.vec_weight);

        // 4. Return top-k as ContextResult
        Ok(to_context_result(fused.into_iter().take(self.top_k).collect()))
    }

    async fn on_turn_complete(&self, _sid: &str, prompt: &str, response: &str) -> Result<()> {
        // Auto-store the assistant response as an episodic memory
        let item = MemoryItem::new(format!("Q: {prompt}\nA: {response}"))
            .with_type(MemoryType::Episodic)
            .with_importance(0.4);
        self.store(item).await
    }
}
```

### 2.4 Result fusion algorithm

**Reciprocal Rank Fusion (RRF):**

```
score(d) = Σ_i  weight_i / (k + rank_i(d))
```

- `k = 60` (standard constant that dampens high-rank outliers)
- FTS5 results ranked by BM25 score
- Vector results ranked by cosine distance

RRF is robust, parameter-free, and produces better results than simple score concatenation when the two distributions are not calibrated against each other.

---

## Phase 3 — Smart injection configuration

**Goal:** Make the memory retrieval behaviour configurable without code changes.

### 3.1 Config schema (HCL)

Add a `memory` block to `safeclaw.hcl` / `policy.hcl`:

```hcl
memory {
  backend = "sqlite"           # "sqlite" | "file" | "in_memory"
  dir     = "~/.config/safeclaw/memory"

  retrieval {
    top_k        = 5
    fts_weight   = 0.4
    vec_weight   = 0.6
    min_score    = 0.2         # discard results below this threshold
    max_tokens   = 800         # token budget for injected context
  }

  embedding {
    provider  = "openai"       # "openai" | "local" | "none"
    model     = "text-embedding-3-small"
    dimension = 1536
  }

  auto_store {
    enabled    = true
    importance = 0.4           # default importance for auto-stored turn memories
  }
}
```

### 3.2 Rust config types

```rust
// crates/safeclaw/src/config.rs  (additions)
pub struct MemoryConfig {
    pub backend: MemoryBackend,
    pub dir: PathBuf,
    pub retrieval: RetrievalConfig,
    pub embedding: EmbeddingConfig,
    pub auto_store: AutoStoreConfig,
}

pub enum MemoryBackend { Sqlite, File, InMemory }

pub struct RetrievalConfig {
    pub top_k: usize,          // default 5
    pub fts_weight: f32,       // default 0.4
    pub vec_weight: f32,       // default 0.6
    pub min_score: f32,        // default 0.2
    pub max_tokens: usize,     // default 800
}

pub struct EmbeddingConfig {
    pub provider: EmbeddingBackend,
    pub model: String,
    pub dimension: usize,
}

pub enum EmbeddingBackend { OpenAi, Local, None }
```

### 3.3 Bootstrap wiring

```rust
// crates/safeclaw/src/bootstrap.rs
pub async fn build_memory_context_provider(
    cfg: &MemoryConfig,
    llm_base_url: &str,
    api_key: &str,
) -> Result<Arc<dyn ContextProvider>> {
    let store = Arc::new(SqliteMemoryStore::new(&cfg.dir).await?);
    let embedder = match cfg.embedding.provider {
        EmbeddingBackend::OpenAi => Some(Arc::new(
            OpenAiEmbeddingProvider::new(llm_base_url, api_key, &cfg.embedding.model)
        ) as Arc<dyn EmbeddingProvider>),
        _ => None,
    };
    Ok(Arc::new(SqliteMemoryContextProvider {
        store,
        embedder,
        fts_weight: cfg.retrieval.fts_weight,
        vec_weight: cfg.retrieval.vec_weight,
        top_k: cfg.retrieval.top_k,
    }))
}
```

---

## Phase 4 — Observability

**Goal:** Surface memory retrieval activity in logs and the SafeClaw UI.

### 4.1 Session log integration

Each retrieval call logs to `session_log`:

```json
{ "event_type": "memory_retrieved", "data": { "query": "...", "hits": 3, "fts": 2, "vec": 1 } }
{ "event_type": "memory_stored",    "data": { "id": "...", "importance": 0.4, "type": "episodic" } }
```

### 4.2 Structured tracing spans

```rust
let _span = tracing::info_span!("memory_retrieval",
    query = %q.query,
    top_k = self.top_k,
).entered();
```

### 4.3 API endpoint (optional)

```
GET /api/memory/search?q=<query>&limit=5
→ { items: [...], fts_count: N, vec_count: M }

GET /api/memory/sessions/<id>/log
→ [{ event_type, data, timestamp_ms }, ...]
```

---

## Implementation order

```
Phase 1  ──►  Phase 2.1  ──►  Phase 2.2  ──►  Phase 2.3  ──►  Phase 3  ──►  Phase 4
  Wire           HTTP           Embedding        Hybrid          Config       Logging
 SQLite       Embedder         Wrapper          Provider         HCL          & API
 backend      (OpenAI)        (a3s-code)       (a3s-code)      (safeclaw)
```

Phase 1 is a standalone shippable improvement. Each subsequent phase adds capability without breaking the previous one.

---

## Files to create / modify

### `crates/memory` (a3s-memory)

| File | Action | Notes |
|------|--------|-------|
| `src/sqlite/mod.rs` | ✅ Done | `SqliteMemoryStore`, `store_with_embedding`, `search_semantic` |
| `src/sqlite/schema.rs` | ✅ Done | DDL, FTS5 triggers, session_log, vec0 table |
| `src/sqlite/markdown.rs` | ✅ Done | Dual-track Markdown export |

### `crates/code` (a3s-code)

| File | Action | Notes |
|------|--------|-------|
| `core/src/memory/embedding_store.rs` | **Create** | `EmbeddingMemoryStore<E>` wrapper |
| `core/src/memory/sqlite_context_provider.rs` | **Create** | `SqliteMemoryContextProvider` with RRF fusion |
| `core/src/memory/rrf.rs` | **Create** | Reciprocal Rank Fusion helper |
| `core/src/memory/mod.rs` | **Modify** | Re-export new types |

### `crates/safeclaw` (SafeClaw gateway)

| File | Action | Notes |
|------|--------|-------|
| `src/config.rs` | **Modify** | Add `MemoryConfig` struct + HCL fields |
| `src/bootstrap.rs` | **Modify** | `init_memory_store()` → `SqliteMemoryStore`; `build_memory_context_provider()` |
| `Cargo.toml` | **Modify** | `a3s-memory = { features = ["sqlite", "sqlite-vec"] }` |

---

## Open questions

1. **Embedding dimension mismatch** — If a user switches embedding models, existing vectors become incompatible. Strategy options: (a) store dimension in DB and reject mismatches, (b) re-embed on model change, (c) keep vectors per-model in separate tables.

2. **Embedding latency** — Calling an HTTP embedding API on every `store()` adds latency to memory writes. Options: (a) async background embedding queue, (b) batch on idle, (c) only embed on explicit `store_with_embedding()` call.

3. **Token budget enforcement** — `max_tokens` in retrieval config needs a token counter. Which tokenizer? Options: (a) character-based approximation (÷ 4), (b) `tiktoken` via WASM, (c) model-specific count from LLM response headers.

4. **FTS5 query sanitisation** — User prompts may contain FTS5 special syntax (`AND`, `OR`, `"`, `*`). The `search()` implementation should escape or use the `columnfilter` syntax to avoid parse errors on adversarial input.
