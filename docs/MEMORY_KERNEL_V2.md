# A3S Memory Kernel V2

Status: Accepted for implementation

## Objective

The memory kernel exists to improve future agent decisions without weakening
correctness, isolation, auditability, or bounded execution. Storing more text is
not a goal by itself.

The first consumer is `a3s-code`, but the contract is runtime-neutral. The
kernel must not depend on an LLM, an agent loop, a session implementation, or a
transport.

## Ownership

`a3s-memory` owns the policy-free integrity boundary:

- namespace validation and isolation;
- typed memory nodes, evidence references, relations, and lifecycle states;
- idempotent and revision-checked atomic changes;
- pure reads and queries;
- complete bounded namespace snapshots for rebuilding caller-owned projections;
- optional exact-namespace change tokens for suppressing redundant snapshots;
- explicit admission and use records;
- backend conformance and rebuildable derived indexes.

The host runtime owns semantic policy:

- deciding when to extract a candidate;
- deciding whether evidence is explicit, verified, or inferred;
- proposing activation, refinement, correction, or supersession;
- generating embeddings and fusing retrieval branches;
- admitting results into a model context;
- scheduling consolidation and retention work;
- exposing CLI, HTTP, MCP, or SDK surfaces.

For `a3s-code`, immutable session, run, artifact, trace, and verification data
remain the source evidence. The memory kernel stores typed references to that
evidence rather than copying the complete session model.

## Required Invariants

### Namespace isolation

Every node, relation, query, change set, and use record belongs to exactly one
`MemoryNamespace`. A namespace contains opaque tenant, principal, and scope
identifiers. Empty identifiers are invalid.

Repository operations never infer or widen a namespace. Cross-namespace
relations are invalid. A caller that is authorized to search more than one
scope issues one explicit query per namespace and performs policy-aware fusion
outside the repository.

### Evidence before activation

Every active derived node has at least one `EvidenceRef`. Evidence includes an
opaque source URI, a content digest, an evidence kind, and the time at which the
source event occurred.

LLM confidence and importance values are annotations, not evidence. A runtime
may create inferred material as `Candidate`, but only an explicit lifecycle
change carrying new Manual or Verification decision evidence may make it
`Active`. The activation revision preserves both the proposal evidence and the
separate decision evidence.

### Non-destructive evolution

Consolidation never physically deletes history. Correction and supersession
create a new revision and preserve prior content, evidence, and relationships.
The active view hides superseded and tombstoned nodes by default.

Physical purge is a separate administrative operation for explicit user or
compliance deletion. It is not a consolidation action.

### Deterministic replay

Callers supply node IDs, change-set idempotency keys, source timestamps, and use
event IDs. Reapplying an identical change set returns its original result.
Reusing an idempotency key with different content fails.

The repository does not call the wall clock or generate durable identities
while applying a change set.

### Optimistic concurrency

Mutations carry an expected node revision. A stale expected revision fails the
entire change set. Multi-node changes are atomic so a replacement node and the
node it supersedes cannot be published independently.

### Pure retrieval

`get` and `query` do not update access counters, timestamps, relevance, or
retention state. Retrieval observation is split into explicit events:

- `record_admission`: the host admitted a node into model context;
- `record_use`: the node was cited, selected, or otherwise used by the host.

Candidate generation, retrieval, admission, and use therefore remain distinct
and auditable. Each admission or use event names the exact node revision that
was observed; the repository rejects references to revisions that do not exist.
Admission accepts only the current `Active` revision, preventing a stale or
concurrently superseded snapshot from entering a new model context. Use may
still cite a historical revision for later audit.

`snapshot_namespace` is also a pure read. It captures one exact namespace and
caller-selected status set under hard node and canonical-payload byte budgets,
sorts nodes by stable ID, and streams a domain-separated SHA-256 identity for
the complete selected view without first allocating the encoded payload. It
never truncates. This gives hosts a deterministic source for rebuilding derived
lexical, vector, or human-readable projections without making those projections
authoritative. Custom backends construct responses through
`MemoryNamespaceSnapshot::try_new`; consumers crossing a backend boundary call
`verify` with the original request before trusting the snapshot identity.

`namespace_change_token` is a separate optional pure read. A backend returning
`Some` promises that every novel successful `apply` changing node state in
the exact namespace publishes a different token at the same linearization
point. Equal tokens therefore prove that the namespace did not change between
the two reads. Sequences may jump but never repeat after a change. Idempotent
replay, failed changes, admission, use, and reads do not advance them.
Persistent backends retain or deterministically reconstruct the same token
after restart. The token is content-free and scoped by the method call; it is
not transferable across namespaces, backend histories, or repository
instances.

The default implementation validates the namespace and returns `None`.
Consumers then retain the complete snapshot path. A token is only a
change-detection accelerator: it is not a snapshot, source-completeness proof
for an initial build, vector-index precondition, lock, distributed lease, or
backend identity.

The caller-owned `VectorIndex` exposes its mutation consistency separately from
repository truth. `partition_atomic` guarantees that searches never observe a
partially replaced partition. `index_revision_cas` additionally compares an
expected global index revision and publishes replacement or removal at the same
linearization point. This prevents delayed derived-index writers and cleanup
tasks from overwriting a newer generation when every writer uses the
conditional API. The in-memory reference index implements this contract;
custom backends default to source-compatible atomic replacement and reject
conditional mutation until they implement it. CAS is intentionally global and
may reject on unrelated partition churn. It is not a distributed lease, a
durable remote backend, or permission to treat vectors as authoritative.

`VectorIndex::change_token` is a separate optional continuity proof. A `Some`
token binds one opaque index-history identity to the exact global revision.
Every effective content mutation must advance that revision, while independent
construction, divergent restore, or rollback must use a different history
identity. This closes the ambiguity where two unrelated indexes can expose the
same revision, record count, and byte count while containing different vectors.
The in-memory index keeps one identity across clones and assigns a new identity
at construction. Custom backends return `None`; a durable backend may retain an
identity across process restarts only when its storage protocol preserves the
same linear history. The token is content-free and is not a vector snapshot,
lease, fencing authority, or durability proof by itself.

### Bounded operations

Queries have validated finite limits. Change sets have a finite operation cap.
Node content, evidence, relations, labels, and identifiers have explicit size
limits. Namespace snapshots have caller-selected node and byte limits beneath
kernel hard caps and fail when the complete selected view does not fit. A
backend must reject over-budget input before changing state, cloning an
over-budget built-in view, or returning a partial view.

## Core Model

The additive V2 API lives in a new `repository` module. The existing
`MemoryItem` and `MemoryStore` API remains available during migration.

### MemoryNamespace

An exact security and ownership partition:

- `tenant_id`
- `principal_id`
- `scope_id`

The scope is opaque to the kernel. Examples include a user scope, a repository
digest, or an agent-specific scope.

### EvidenceRef

A reference to immutable evidence:

- `uri`
- `digest`
- `kind`: `UserStatement`, `ToolResult`, `Artifact`, `SessionTurn`,
  `Verification`, or `Manual`
- `occurred_at`

The kernel validates shape and bounds but does not resolve the URI or verify the
digest. That responsibility belongs to the authorized host.

### MemoryNode

A durable node contains:

- caller-supplied ID and exact namespace;
- `Episodic`, `Semantic`, or `Procedural` kind;
- `Candidate`, `Active`, `Superseded`, `Conflicted`, or `Tombstoned` status;
- content, confidence, importance, evidence, typed relations, and labels;
- monotonically increasing revision;
- caller-supplied creation and update timestamps.

Working and short-term memory remain runtime state in `a3s-code`; they are not
durable V2 node kinds.

### MemoryRelation

Relations are typed and validated, not comma-separated metadata:

- `Supersedes`
- `SupersededBy`
- `ConflictsWith`
- `RelatedTo`

Targets must exist in the same namespace at commit time. Symmetric or inverse
edges are represented explicitly in the same atomic change set.

### MemoryChangeSet

A change set contains an idempotency key, namespace, occurrence time, and a
bounded sequence of operations. Initial operations are:

- create a candidate or active node;
- activate a candidate with new decision evidence;
- corroborate with new evidence;
- refine or correct content with new evidence;
- add or remove a relation;
- change status to superseded, conflicted, or tombstoned.

Each mutation names the expected revision. The repository validates all
operations against a staged state and publishes either the complete result or
nothing.

### MemoryQuery

A query is scoped to one exact namespace and can filter by text, kinds, and
statuses. The default status filter is `Active`. Results contain immutable node
snapshots plus retrieval score details. Queries are pure.

The reference repository implements the versioned deterministic lexical
profile exposed as `MEMORY_LEXICAL_QUERY_PROFILE_V1`. It preserves lowercased
alphanumeric words and adds the complete span plus overlapping character
bigrams for contiguous Chinese, Japanese, and Korean runs. Bigrams recover
shared same-language phrases without the false-positive pressure of CJK
unigrams. They do not infer translated or no-overlap semantic equivalence.
Vector retrieval remains caller-owned until an evaluation proves that it
improves the target workload. The existing `VectorIndex` remains available
independently.

## Storage and Projection

V2 defines behavior, not a mandatory physical representation.

The in-memory implementation is the executable reference contract. The local
file backend uses a checksummed, single-writer write-ahead journal and runs the
same conformance suite. It syncs validated records before publishing them,
replays idempotently after restart, truncates only an incomplete final record,
and fails closed on committed-record corruption. SQLite, JSON snapshots,
keyword indexes, vector indexes, relation graphs, and Markdown remain backend
choices or derived projections.

Markdown is not made authoritative in the first release. A deterministic
human-readable projection may be rebuilt from repository state. Bidirectional
editing requires a later design with revision preconditions and conflict
resolution; append-only Markdown logs must not be presented as the current
active state.

## Compatibility

V1 remains source-compatible while V2 is introduced:

- existing `MemoryStore` implementations continue to compile;
- existing `MemoryRepository` implementations default to no change token;
- existing `VectorIndex` implementations default to no change token;
- existing `MemoryItem` serialization remains unchanged;
- V2 uses new types rather than interpreting V1 string metadata as trusted
  typed fields;
- `a3s-code` moves to V2 behind an explicit option and initially runs in shadow
  mode;
- a migration tool may convert eligible V1 items into V2 candidates, but no V1
  item becomes active without evidence and namespace assignment.

V1 behavior that mutates access counters during reads remains unchanged for
compatibility. V2 must not repeat that behavior.

## Implementation Sequence

1. Add the V2 types, validation errors, and repository trait.
2. Write a reusable backend conformance suite before implementing storage.
3. Implement the in-memory reference repository with atomic change sets.
4. Add a persistent backend only after crash, replay, and concurrency contracts
   are executable.
5. Integrate `a3s-code` in candidate-only shadow mode with typed evidence.
6. Enable active recall only after isolation, evidence, and quality gates pass.
7. Add consolidation through an owned host or `a3s-flow` lifecycle.
8. Evaluate lexical retrieval before adding vector fusion or graph expansion.
9. Add human projections and proactive behavior last.

## Release Gates

The V2 kernel is not complete until tests prove:

- zero cross-namespace reads, writes, deduplication, or relations;
- every active derived node has proposal and activation-decision evidence;
- identical replay is idempotent and conflicting replay is rejected;
- stale revisions cannot partially apply;
- correction and supersession preserve prior revisions;
- query operations leave repository state unchanged;
- namespace snapshots are complete, deterministically ordered and hashed, and
  reject node- or byte-over-budget views without truncation across every
  conforming backend;
- built-in namespace change tokens advance atomically for novel successful
  changes, remain stable for replay/failure/access, stay namespace-isolated,
  reject an unsupported profile, and reconstruct exactly after file restart;
- built-in vector change tokens bind one clone-shared history and revision,
  advance on effective mutation, remain stable on no-op mutation, reject
  malformed serialized evidence, and differ across independent indexes even
  when their logical status collides;
- the shared in-memory/file contract retrieves partial CJK phrases under the
  exact versioned lexical profile without adding single-character matching;
- admission and use events are independently idempotent, and stale/inactive
  revisions cannot be newly admitted;
- invalid and over-budget inputs leave state unchanged;
- concurrent writers produce one valid serializable result;
- persistent backends recover the last committed state after interruption.

The product integration additionally requires an evaluation against no-memory
and V1 baselines for write precision, evidence fidelity, conflict handling,
task success, context tokens, latency, and model cost.
