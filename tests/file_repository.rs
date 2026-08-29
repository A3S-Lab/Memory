use a3s_memory::repository::{
    DurableMemoryKind, EvidenceKind, EvidenceRef, FileMemoryRepository, MemoryAccessEvent,
    MemoryChangeSet, MemoryNamespace, MemoryNodeDraft, MemoryOperation, MemoryRepository,
    MemoryRepositoryError, MemorySnapshotRequest, MemoryStatus, RevisionMode,
};
use chrono::{TimeZone, Utc};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;

#[path = "support/repository_contract.rs"]
mod repository_contract;

fn time(second: u32) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 29, 14, 0, second)
        .single()
        .expect("valid test time")
}

fn namespace() -> MemoryNamespace {
    MemoryNamespace::try_new("tenant-a", "principal-a", "repo-a").unwrap()
}

fn evidence(name: &str, second: u32) -> EvidenceRef {
    EvidenceRef::try_new(
        format!("a3s://session/run/turn/{name}"),
        format!("sha256:{name:0>64}"),
        EvidenceKind::SessionTurn,
        time(second),
    )
    .unwrap()
}

fn create_change(namespace: &MemoryNamespace) -> MemoryChangeSet {
    MemoryChangeSet::new(
        "create-node",
        namespace.clone(),
        time(1),
        vec![MemoryOperation::Create {
            node: MemoryNodeDraft::new(
                "node",
                namespace.clone(),
                DurableMemoryKind::Semantic,
                MemoryStatus::Active,
                "persisted memory",
                vec![evidence("create", 1)],
                time(1),
            ),
        }],
    )
}

#[tokio::test]
async fn file_backend_passes_the_reusable_repository_contract() {
    let directory = tempfile::tempdir().unwrap();
    let repository = FileMemoryRepository::open(directory.path()).await.unwrap();
    repository_contract::assert_repository_contract(&repository, "file-contract").await;
}

#[tokio::test]
async fn restart_preserves_idempotency_history_and_usage_events() {
    let directory = tempfile::tempdir().unwrap();
    let namespace = namespace();
    let create = create_change(&namespace);
    let first_result;
    let snapshot_digest;
    {
        let repository = FileMemoryRepository::open(directory.path()).await.unwrap();
        first_result = repository.apply(create.clone()).await.unwrap();
        repository
            .record_admission(MemoryAccessEvent::new(
                "admission",
                namespace.clone(),
                "node",
                1,
                time(2),
            ))
            .await
            .unwrap();
        repository
            .apply(MemoryChangeSet::new(
                "revise-node",
                namespace.clone(),
                time(3),
                vec![MemoryOperation::Revise {
                    node_id: "node".into(),
                    expected_revision: 1,
                    content: "corrected persisted memory".into(),
                    mode: RevisionMode::Correction,
                    evidence: vec![evidence("correction", 3)],
                    confidence: None,
                    importance: None,
                }],
            ))
            .await
            .unwrap();
        snapshot_digest = repository
            .snapshot_namespace(MemorySnapshotRequest::new(namespace.clone(), 4))
            .await
            .unwrap()
            .digest;
    }

    let reopened = FileMemoryRepository::open(directory.path()).await.unwrap();
    let journal = directory.path().join("memory-v2.journal");
    let before_replay = std::fs::metadata(&journal).unwrap().len();
    assert_eq!(reopened.apply(create).await.unwrap(), first_result);
    assert_eq!(std::fs::metadata(&journal).unwrap().len(), before_replay);
    let node = reopened.get(&namespace, "node").await.unwrap().unwrap();
    assert_eq!(node.revision, 2);
    assert_eq!(node.history[0].content, "persisted memory");
    assert_eq!(
        reopened
            .snapshot_namespace(MemorySnapshotRequest::new(namespace.clone(), 4))
            .await
            .unwrap()
            .digest,
        snapshot_digest
    );
    let usage = reopened.usage_summary(&namespace, "node").await.unwrap();
    assert_eq!(usage.admissions, 1);
    assert_eq!(usage.uses, 0);
}

#[tokio::test]
async fn failed_preflight_does_not_append_or_partially_publish() {
    let directory = tempfile::tempdir().unwrap();
    let namespace = namespace();
    let repository = FileMemoryRepository::open(directory.path()).await.unwrap();
    repository.apply(create_change(&namespace)).await.unwrap();
    let journal = directory.path().join("memory-v2.journal");
    let committed_length = std::fs::metadata(&journal).unwrap().len();

    let error = repository
        .apply(MemoryChangeSet::new(
            "invalid-atomic-change",
            namespace.clone(),
            time(2),
            vec![
                MemoryOperation::Revise {
                    node_id: "node".into(),
                    expected_revision: 1,
                    content: "must not be published".into(),
                    mode: RevisionMode::Correction,
                    evidence: vec![evidence("first-staged", 2)],
                    confidence: None,
                    importance: None,
                },
                MemoryOperation::Revise {
                    node_id: "node".into(),
                    expected_revision: 99,
                    content: "stale update".into(),
                    mode: RevisionMode::Correction,
                    evidence: vec![evidence("stale", 2)],
                    confidence: None,
                    importance: None,
                },
            ],
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        MemoryRepositoryError::RevisionConflict { .. }
    ));
    assert_eq!(std::fs::metadata(&journal).unwrap().len(), committed_length);
    let node = repository.get(&namespace, "node").await.unwrap().unwrap();
    assert_eq!(node.revision, 1);
    assert_eq!(node.content, "persisted memory");
}

#[tokio::test]
async fn concurrent_writers_persist_one_serializable_winner() {
    let directory = tempfile::tempdir().unwrap();
    let namespace = namespace();
    let repository = Arc::new(FileMemoryRepository::open(directory.path()).await.unwrap());
    repository.apply(create_change(&namespace)).await.unwrap();

    let update = |key: &str, content: &str| {
        MemoryChangeSet::new(
            key,
            namespace.clone(),
            time(2),
            vec![MemoryOperation::Revise {
                node_id: "node".into(),
                expected_revision: 1,
                content: content.into(),
                mode: RevisionMode::Correction,
                evidence: vec![evidence(key, 2)],
                confidence: None,
                importance: None,
            }],
        )
    };
    let left_repository = repository.clone();
    let right_repository = repository.clone();
    let (left, right) = tokio::join!(
        left_repository.apply(update("left", "left persisted")),
        right_repository.apply(update("right", "right persisted")),
    );
    let winner = match (left, right) {
        (Ok(result), Err(MemoryRepositoryError::RevisionConflict { .. }))
        | (Err(MemoryRepositoryError::RevisionConflict { .. }), Ok(result)) => {
            result.nodes[0].content.clone()
        }
        result => panic!("expected one serializable winner, got {result:?}"),
    };
    drop(left_repository);
    drop(right_repository);
    drop(repository);

    let reopened = FileMemoryRepository::open(directory.path()).await.unwrap();
    let node = reopened.get(&namespace, "node").await.unwrap().unwrap();
    assert_eq!(node.revision, 2);
    assert_eq!(node.content, winner);
}

#[tokio::test]
async fn torn_final_record_is_truncated_before_replay() {
    let directory = tempfile::tempdir().unwrap();
    let namespace = namespace();
    {
        let repository = FileMemoryRepository::open(directory.path()).await.unwrap();
        repository.apply(create_change(&namespace)).await.unwrap();
    }
    let journal = directory.path().join("memory-v2.journal");
    let committed_length = std::fs::metadata(&journal).unwrap().len();
    OpenOptions::new()
        .append(true)
        .open(&journal)
        .unwrap()
        .write_all(b"{\"version\":1")
        .unwrap();
    assert!(std::fs::metadata(&journal).unwrap().len() > committed_length);

    let reopened = FileMemoryRepository::open(directory.path()).await.unwrap();
    assert!(reopened.get(&namespace, "node").await.unwrap().is_some());
    assert_eq!(std::fs::metadata(&journal).unwrap().len(), committed_length);
}

#[tokio::test]
async fn checksum_corruption_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let namespace = namespace();
    {
        let repository = FileMemoryRepository::open(directory.path()).await.unwrap();
        repository.apply(create_change(&namespace)).await.unwrap();
    }
    let journal = directory.path().join("memory-v2.journal");
    let mut bytes = std::fs::read(&journal).unwrap();
    let marker = b"persisted memory";
    let offset = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .expect("journal contains node content");
    bytes[offset] = b'P';
    std::fs::write(&journal, bytes).unwrap();

    let error = FileMemoryRepository::open(directory.path())
        .await
        .unwrap_err();
    assert!(matches!(error, MemoryRepositoryError::Persistence { .. }));
}

#[tokio::test]
async fn a_second_live_writer_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let first = FileMemoryRepository::open(directory.path()).await.unwrap();
    let error = FileMemoryRepository::open(directory.path())
        .await
        .unwrap_err();
    assert!(matches!(error, MemoryRepositoryError::Persistence { .. }));
    drop(first);
    FileMemoryRepository::open(directory.path()).await.unwrap();
}
