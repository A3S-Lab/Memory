# a3s-memory

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

面向 A3S 的可插拔记忆存储。

提供 `MemoryStore` trait 与两种默认实现。需要跨会话持久化并召回知识的 Agent 直接依赖本 crate——无需其他依赖。

本 crate 还提供独立的 `VectorIndex` 能力，用于调用方自有的语义检索。其内存后端无额外依赖；可选的 SQLite 后端在进程重启后保留索引历史与修订 CAS。向量索引不是记忆存储：文档准入、嵌入生成、生命周期与结果融合仍由调用方负责。

默认存储还强制执行少量记忆卫生：规范化后的精确重复项会合并进已有条目，标签/元数据/重要性会合并，剪枝会保护精选记忆（如 pinned、频繁召回、已合并或冲突跟踪项）。存储层不会从关键词重叠推断语义等价。

## 设计

本 crate 遵循最小核心 + 外部扩展模式：

**核心（稳定、不可替换）：**
- `MemoryStore` — 存储后端 trait
- `MemoryItem` — 记忆单元
- `MemoryType` — episodic / semantic / procedural / working
- `RelevanceConfig` — 评分参数

**扩展（可通过 `MemoryStore` 替换）：**
- `InMemoryStore` — 默认、临时（测试与非持久场景）
- `FileMemoryStore` — 持久、原子写入、内存索引

三层会话记忆（`AgentMemory`）与上下文注入（`MemoryContextProvider`）位于 `a3s-code`，不在此处。本 crate 仅拥有存储层。

### 持久仓储 V2

附加的 `repository` 模块为有证据支持的持久记忆提供无策略的完整性内核。它加入精确的租户/主体/作用域命名空间、candidate-to-active 生命周期转换、不可变证据引用、类型化关系、完整修订历史、乐观并发，以及幂等原子变更集。Candidate 激活需要新的决策证据，因此 LLM 标注不能静默成为服务状态。读操作是纯的；准入与使用会针对宿主观察到的精确节点修订显式记录。重建派生投影的宿主可请求完整有界命名空间快照：仓储过滤精确状态集、确定性排序节点、流式传输稳定的 SHA-256 视图身份，并在节点或规范字节超预算时拒绝视图，而不是静默截断。快照响应可相对原始请求重算，因此宿主无需信任自定义后端声称的摘要。

后端还可选择实现 `MemoryRepository::namespace_change_token`。相等的有界 token 证明两次读取之间没有新的成功变更集改动该精确命名空间，使调用方可避免冗余快照，而不削弱其正常快照与发布证明。Token 仅包含版本化单调序列；不含命名空间标识符或记忆内容。`InMemoryRepository` 在与节点状态相同的写锁线性化点更新序列，`FileMemoryRepository` 通过日志重放重建相同序列。幂等重放、失败变更、准入与使用不会推进它。自定义后端返回 `None`，除非显式实现该契约。

确定性 V2 词法配置保留小写字母数字词，并为连续的中日韩文本加入重叠 bigram。这使得普通同语言 CJK 短语变体无需模型或外发即可检索，同时避免嘈杂的单字匹配。宿主可绑定 `MEMORY_LEXICAL_QUERY_PROFILE_V1` 以检测算法漂移。该配置不是跨语言语义检索：翻译或无重叠释义仍需要调用方自有的经评估检索扩展。

`InMemoryRepository` 是可执行参考实现。`FileMemoryRepository` 通过带校验和的预写日志加入本地持久性：经验证的操作在发布前追加并同步，重启后确定性重放，并由单写者目录锁保护。撕裂的最终记录会被丢弃；已提交记录中的校验和损坏会失败闭合。

现有 `MemoryItem` 与 `MemoryStore` API 在宿主迁移到 V2 期间保持源码兼容。抽取、合并策略、嵌入、上下文准入与调度仍由宿主运行时拥有。不变量与发布门见 [`docs/MEMORY_KERNEL_V2.md`](docs/MEMORY_KERNEL_V2.md)。

## 用法

```toml
[dependencies]
a3s-memory = { version = "0.1", path = "../memory" }
```

### 存储与检索

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

### 持久存储

```rust
use a3s_memory::{FileMemoryStore, MemoryStore};

let store = FileMemoryStore::new("/var/lib/agent/memory").await?;
// Directory layout:
//   memory/
//     index.json        ← in-memory index, persisted atomically
//     items/{id}.json   ← one file per memory item
```

### 自定义后端

实现 `MemoryStore` 以使用任意存储系统（SQLite、向量数据库等）：

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

### 调用方自有的向量搜索

`InMemoryVectorIndex` 将调用方提供的向量存储在不可变分区快照中，并执行精确有界 top-k 搜索。它不使用 SQLite、不持久化数据、不调用嵌入模型，也不生成后台任务。最终所有者丢弃时释放索引。

```rust
use a3s_memory::{
    InMemoryVectorIndex, VectorIndex, VectorIndexDescriptor, VectorRecord,
    VectorRevision, VectorSearchRequest,
};

let index = InMemoryVectorIndex::new(
    VectorIndexDescriptor::new(3)
        .with_max_records(10_000)
        .with_max_bytes(64 * 1024 * 1024),
)?;

index
    .replace_partition_if_revision(
        "src/lib.rs",
        VectorRevision::new(0),
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

维度在索引构造时选定。余弦索引在准入时规范化记录与查询，拒绝零/非有限向量，并返回产生每个结果页的不可变索引修订。替换一个分区会原子发布其完整新记录集，同时共享所有未变更的分区块。`InMemoryVectorIndex` 还声明 `index_revision_cas`：条件替换与移除在与发布相同的线性化点比较期望的全局索引修订。因此延迟的写入者会以 `RevisionConflict` 失败，而不是覆盖或删除更新的一代。自定义后端保持源码兼容，并默认 `partition_atomic`；其条件方法在后端实现 CAS 契约前失败闭合。因为前置条件是全局索引修订，不相关的分区变更可能保守地拒绝一次已准备的更新。

后端还可暴露 `VectorIndex::change_token()`，作为单一索引历史与修订的精确连续性证据。`InMemoryVectorIndex` 在构造时分配新鲜的不透明 SHA-256 历史摘要；克隆保留它，每一次有效变更推进 token 修订。因此两个独立构造的索引即使计数器与字节大小碰巧相同，也有不同 token。自定义索引默认返回 `None`。持久后端仅在能证明相同线性变更历史时，才可跨进程重启保留历史摘要；重建、回滚或发散恢复需要新身份。Token 不是内容快照、分布式租约或远程持久性声明。

对正确性敏感的调用方应使用异步 `VectorIndex::observe()` 方法。它返回一个自洽的状态/token 对，并可报告存储失败。同步的 `status()` 与 `change_token()` 仍是兼容视图，对持久或远程实现可能过时。

启用 `sqlite` feature 时，`SqliteVectorIndex::open(path, descriptor)` 加入本地持久的精确搜索后端。它在一个 SQLite 数据库中存储描述符、不透明历史身份、修订、稳定逻辑字节核算、分区完整性摘要与向量行。变更使用 `IMMEDIATE` 事务，因此独立进程共享一个全局修订-CAS 线性化点。重新打开会在提供服务前校验描述符、核算、记录形状与完整性摘要；不匹配与损坏失败闭合。在 Unix 与 Windows 上，历史身份还绑定到数据库文件身份：复制或原子替换文件会在下次打开时分叉 token，而不改变其内容修订。备份恢复必须替换已关闭的数据库文件；不支持原地覆盖与并发带外文件操作。其他目标在无法获得稳定文件身份时，每次打开会保守地分叉 token。阻塞的 SQLite 工作在 Tokio 的 blocking 池上运行。这是本地持久性与围栏，不是分布式租约或远程复制存储。

```rust
use a3s_memory::{SqliteVectorIndex, VectorIndex, VectorIndexDescriptor};

let index = SqliteVectorIndex::open(
    "/var/lib/agent/semantic.sqlite3",
    VectorIndexDescriptor::new(384),
)
.await?;
let observation = index.observe().await?;
```

对 25,000 条、384 维记录运行锁定的发布资格验证：`cargo run --example vector_search_benchmark --release`。它输出 JSON 证据，并在精确 top-20 搜索超过 30 ms p95 预算时失败。

## 相关性评分

搜索将词法匹配强度（精确短语、词项、标签与记忆类型匹配）与下方相关性分数结合。精确或更具体的查询匹配会排在通用高重要性记忆之前，而同等具体的结果仍受益于重要性与近因。

```
score = importance × importance_weight + decay × recency_weight
decay = exp(−age_days / decay_days)
```

默认：`importance_weight = 0.7`，`recency_weight = 0.3`，`decay_days = 30`。

```rust
use a3s_memory::{MemoryItem, RelevanceConfig};

let config = RelevanceConfig {
    decay_days: 7.0,        // faster decay
    importance_weight: 0.9,
    recency_weight: 0.1,
};

let score = item.relevance_score_at(now, &config);
```

## 去重与剪枝

`InMemoryStore`、`FileMemoryStore` 与可选 SQLite 存储在规范化大小写与空白后折叠精确持久重复项。标点仍有意义。第一个记忆 id 保持规范；后续重复项提升重要性，合并标签与列表式元数据（如 `supersedes` / `conflicts_with`），并记录 `duplicate_count` 元数据。

当调用方需要现在代表该事实的规范条目时，使用 `MemoryStore::store_and_return()`。语义合并属于上游模型或有足够上下文做出该判断的调用方。此类调用方可显式使用 `MemoryItem::merge_duplicate()`，或持久化关系元数据并让拥有的记忆运行时应用它。

`PrunePolicy` 移除陈旧、低重要性条目，并可强制最大条目数，但会硬保护精选记忆：`keep` / `pinned` / `protected` 标签或元数据、反复访问的条目，以及携带 `supersedes` / `conflicts_with` 关系元数据的记忆。

## 本 crate 不拥有的内容

| 关注点 | 所在位置 |
|---------|----------|
| 三层会话记忆（working / short-term / long-term） | `a3s-code` |
| `MemoryConfig`（max_short_term, max_working） | `a3s-code` |
| `MemoryStats` | `a3s-code` |
| 向 Agent 提示注入上下文 | `a3s-code` |
| 工作区扫描、代码分块、嵌入与混合排序 | `a3s-code` |

## 测试

测试套件覆盖 `MemoryItem`、`RelevanceConfig`、`InMemoryStore`、`FileMemoryStore`，以及可复用的 V2 仓储契约。两个 V2 后端运行相同行为套件；文件测试额外覆盖重启恢复、单写者锁定、撕裂写入、损坏与持久并发。V2 测试覆盖命名空间隔离、证据准入、幂等重放、原子回滚、修订保留、纯查询、显式使用记录、有界输入、完整命名空间快照身份与溢出行为、跨两个 V2 后端的确定性词/CJK-bigram 检索，以及并发写入者。可选的命名空间 change-token 套件额外覆盖原子推进、命名空间隔离、幂等与失败写入、访问事件稳定性、并发单胜者更新、重启重建、篡改拒绝、脱敏，以及源码兼容的自定义后端选择加入。向量套件证明原子观察、克隆连续性、有效变更推进、无操作稳定性、序列化校验，以及逻辑状态碰撞时独立构造索引的不同历史。启用 `sqlite` feature 额外证明重启连续性、跨连接单胜者 CAS、稳定跨后端字节核算、描述符漂移拒绝、内容完整性检查与失败闭合的损坏处理，以及 SQLite V1 存储契约。`sqlite-vec` 门另外证明并发首次连接在任一 SQLite 连接打开前注册 `vec0` 自动扩展。

```sh
cargo test
```

## 社区

加入我们的 [Discord](https://discord.gg/XVg6Hu6H) 以获取问题、讨论与更新。

## 许可证

MIT
