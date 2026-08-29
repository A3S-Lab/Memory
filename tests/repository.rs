use a3s_memory::repository::{
    DurableMemoryKind, EvidenceKind, EvidenceRef, InMemoryRepository, MemoryAccessEvent,
    MemoryChangeSet, MemoryNamespace, MemoryNodeDraft, MemoryOperation, MemoryQuery,
    MemoryRelation, MemoryRelationKind, MemoryRepository, MemoryRepositoryError, MemoryStatus,
    RevisionMode, MAX_CONTENT_BYTES,
};
use chrono::{DateTime, TimeZone, Utc};
use std::sync::Arc;

#[path = "support/repository_contract.rs"]
mod repository_contract;

fn time(second: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 29, 12, 0, second)
        .single()
        .expect("valid test time")
}

fn namespace(scope: &str) -> MemoryNamespace {
    MemoryNamespace::try_new("tenant-a", "principal-a", scope).expect("valid namespace")
}

fn evidence(name: &str, second: u32) -> EvidenceRef {
    EvidenceRef::try_new(
        format!("a3s://session/run-1/turn/{name}"),
        format!("sha256:{name:0>64}"),
        EvidenceKind::SessionTurn,
        time(second),
    )
    .expect("valid evidence")
}

fn draft(
    namespace: MemoryNamespace,
    id: &str,
    status: MemoryStatus,
    content: &str,
    second: u32,
) -> MemoryNodeDraft {
    MemoryNodeDraft::new(
        id,
        namespace,
        DurableMemoryKind::Semantic,
        status,
        content,
        vec![evidence(id, second)],
        time(second),
    )
}

fn changes(
    namespace: MemoryNamespace,
    key: &str,
    second: u32,
    operations: Vec<MemoryOperation>,
) -> MemoryChangeSet {
    MemoryChangeSet::new(key, namespace, time(second), operations)
}

#[tokio::test]
async fn in_memory_backend_passes_the_reusable_repository_contract() {
    let repository = InMemoryRepository::new();
    repository_contract::assert_repository_contract(&repository, "in-memory-contract").await;
}

#[tokio::test]
async fn namespace_is_exact_for_ids_queries_and_relations() {
    let repository = InMemoryRepository::new();
    let repo_a = namespace("repo-a");
    let repo_b = namespace("repo-b");

    for (key, ns, content) in [
        ("create-a", repo_a.clone(), "alpha memory"),
        ("create-b", repo_b.clone(), "beta memory"),
    ] {
        repository
            .apply(changes(
                ns.clone(),
                key,
                1,
                vec![MemoryOperation::Create {
                    node: draft(ns, "same-id", MemoryStatus::Active, content, 1),
                }],
            ))
            .await
            .unwrap();
    }

    assert_eq!(
        repository
            .get(&repo_a, "same-id")
            .await
            .unwrap()
            .unwrap()
            .content,
        "alpha memory"
    );
    assert_eq!(
        repository
            .get(&repo_b, "same-id")
            .await
            .unwrap()
            .unwrap()
            .content,
        "beta memory"
    );

    let query = repository
        .query(MemoryQuery::new(repo_a.clone()).with_text("beta"))
        .await
        .unwrap();
    assert!(query.hits.is_empty());

    let error = repository
        .apply(changes(
            repo_a.clone(),
            "foreign-relation",
            2,
            vec![MemoryOperation::AddRelation {
                node_id: "same-id".into(),
                expected_revision: 1,
                relation: MemoryRelation::new(MemoryRelationKind::RelatedTo, "only-in-b"),
            }],
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        MemoryRepositoryError::RelationTargetNotFound { .. }
    ));
}

#[tokio::test]
async fn active_nodes_require_evidence_and_candidates_need_explicit_activation() {
    let repository = InMemoryRepository::new();
    let ns = namespace("repo-a");
    let mut active = draft(
        ns.clone(),
        "unsupported",
        MemoryStatus::Active,
        "unsupported claim",
        1,
    );
    active.evidence.clear();

    let error = repository
        .apply(changes(
            ns.clone(),
            "unsupported",
            1,
            vec![MemoryOperation::Create { node: active }],
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        MemoryRepositoryError::EvidenceRequired { .. }
    ));

    repository
        .apply(changes(
            ns.clone(),
            "candidate",
            2,
            vec![MemoryOperation::Create {
                node: draft(
                    ns.clone(),
                    "candidate",
                    MemoryStatus::Candidate,
                    "candidate claim",
                    2,
                ),
            }],
        ))
        .await
        .unwrap();
    assert!(repository
        .query(MemoryQuery::new(ns.clone()))
        .await
        .unwrap()
        .hits
        .is_empty());

    repository
        .apply(changes(
            ns.clone(),
            "activate",
            3,
            vec![MemoryOperation::Activate {
                node_id: "candidate".into(),
                expected_revision: 1,
                evidence: vec![EvidenceRef::try_new(
                    "a3s://review/activation/candidate",
                    format!("sha256:{:0>64}", "activation"),
                    EvidenceKind::Verification,
                    time(3),
                )
                .unwrap()],
            }],
        ))
        .await
        .unwrap();
    let active = repository.get(&ns, "candidate").await.unwrap().unwrap();
    assert_eq!(active.evidence.len(), 2);
    assert!(active
        .evidence
        .iter()
        .any(|evidence| evidence.kind == EvidenceKind::Verification));
    assert_eq!(
        repository
            .query(MemoryQuery::new(ns.clone()))
            .await
            .unwrap()
            .hits
            .len(),
        1
    );

    let missing_decision_evidence = repository
        .apply(changes(
            ns,
            "activate-without-decision-evidence",
            4,
            vec![MemoryOperation::Activate {
                node_id: "candidate".into(),
                expected_revision: 2,
                evidence: Vec::new(),
            }],
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        missing_decision_evidence,
        MemoryRepositoryError::EvidenceRequired { .. }
    ));
}

#[tokio::test]
async fn identical_replay_is_idempotent_and_conflicting_replay_is_rejected() {
    let repository = InMemoryRepository::new();
    let ns = namespace("repo-a");
    let change_set = changes(
        ns.clone(),
        "stable-key",
        1,
        vec![MemoryOperation::Create {
            node: draft(ns.clone(), "one", MemoryStatus::Active, "first value", 1),
        }],
    );

    let first = repository.apply(change_set.clone()).await.unwrap();
    let replay = repository.apply(change_set).await.unwrap();
    assert_eq!(first, replay);

    let error = repository
        .apply(changes(
            ns.clone(),
            "stable-key",
            1,
            vec![MemoryOperation::Create {
                node: draft(ns, "two", MemoryStatus::Active, "different value", 1),
            }],
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        MemoryRepositoryError::IdempotencyConflict { .. }
    ));
}

#[tokio::test]
async fn stale_revision_rolls_back_the_entire_change_set() {
    let repository = InMemoryRepository::new();
    let ns = namespace("repo-a");
    repository
        .apply(changes(
            ns.clone(),
            "seed",
            1,
            vec![
                MemoryOperation::Create {
                    node: draft(ns.clone(), "one", MemoryStatus::Active, "one before", 1),
                },
                MemoryOperation::Create {
                    node: draft(ns.clone(), "two", MemoryStatus::Active, "two before", 1),
                },
            ],
        ))
        .await
        .unwrap();

    let error = repository
        .apply(changes(
            ns.clone(),
            "atomic-update",
            2,
            vec![
                MemoryOperation::Revise {
                    node_id: "one".into(),
                    expected_revision: 1,
                    content: "one after".into(),
                    mode: RevisionMode::Refinement,
                    evidence: vec![evidence("one-refined", 2)],
                    confidence: None,
                    importance: None,
                },
                MemoryOperation::Revise {
                    node_id: "two".into(),
                    expected_revision: 99,
                    content: "two after".into(),
                    mode: RevisionMode::Correction,
                    evidence: vec![evidence("two-corrected", 2)],
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

    for (id, content) in [("one", "one before"), ("two", "two before")] {
        let node = repository.get(&ns, id).await.unwrap().unwrap();
        assert_eq!(node.revision, 1);
        assert_eq!(node.content, content);
        assert!(node.history.is_empty());
    }
}

#[tokio::test]
async fn correction_and_supersession_preserve_revision_history() {
    let repository = InMemoryRepository::new();
    let ns = namespace("repo-a");
    repository
        .apply(changes(
            ns.clone(),
            "seed-old",
            1,
            vec![MemoryOperation::Create {
                node: draft(
                    ns.clone(),
                    "old",
                    MemoryStatus::Active,
                    "the port is 3000",
                    1,
                ),
            }],
        ))
        .await
        .unwrap();
    repository
        .apply(changes(
            ns.clone(),
            "correct-old",
            2,
            vec![MemoryOperation::Revise {
                node_id: "old".into(),
                expected_revision: 1,
                content: "the port was 3000".into(),
                mode: RevisionMode::Correction,
                evidence: vec![evidence("old-correction", 2)],
                confidence: Some(0.9),
                importance: None,
            }],
        ))
        .await
        .unwrap();

    repository
        .apply(changes(
            ns.clone(),
            "supersede",
            3,
            vec![
                MemoryOperation::Create {
                    node: draft(
                        ns.clone(),
                        "new",
                        MemoryStatus::Active,
                        "the port is 4000",
                        3,
                    ),
                },
                MemoryOperation::AddRelation {
                    node_id: "old".into(),
                    expected_revision: 2,
                    relation: MemoryRelation::new(MemoryRelationKind::SupersededBy, "new"),
                },
                MemoryOperation::AddRelation {
                    node_id: "new".into(),
                    expected_revision: 1,
                    relation: MemoryRelation::new(MemoryRelationKind::Supersedes, "old"),
                },
                MemoryOperation::SetStatus {
                    node_id: "old".into(),
                    expected_revision: 3,
                    status: MemoryStatus::Superseded,
                },
            ],
        ))
        .await
        .unwrap();

    let old = repository.get(&ns, "old").await.unwrap().unwrap();
    assert_eq!(old.status, MemoryStatus::Superseded);
    assert_eq!(old.revision, 4);
    assert_eq!(old.history.len(), 3);
    assert_eq!(old.history[0].content, "the port is 3000");
    assert_eq!(old.history[1].content, "the port was 3000");
    assert_eq!(old.history[2].status, MemoryStatus::Active);

    let frozen_error = repository
        .apply(changes(
            ns.clone(),
            "rewrite-frozen",
            4,
            vec![MemoryOperation::Revise {
                node_id: "old".into(),
                expected_revision: 4,
                content: "silently rewritten".into(),
                mode: RevisionMode::Correction,
                evidence: vec![evidence("rewrite-frozen", 4)],
                confidence: None,
                importance: None,
            }],
        ))
        .await
        .unwrap_err();
    assert!(matches!(
        frozen_error,
        MemoryRepositoryError::InvariantViolation { .. }
    ));
    assert_eq!(
        repository.get(&ns, "old").await.unwrap().unwrap().revision,
        4
    );

    let active = repository.query(MemoryQuery::new(ns)).await.unwrap();
    assert_eq!(active.hits.len(), 1);
    assert_eq!(active.hits[0].node.id, "new");
}

#[tokio::test]
async fn query_is_pure_and_scores_lexical_matches_deterministically() {
    let repository = InMemoryRepository::new();
    let ns = namespace("repo-a");
    repository
        .apply(changes(
            ns.clone(),
            "seed",
            1,
            vec![
                MemoryOperation::Create {
                    node: draft(
                        ns.clone(),
                        "exact",
                        MemoryStatus::Active,
                        "rust memory kernel",
                        1,
                    ),
                },
                MemoryOperation::Create {
                    node: draft(
                        ns.clone(),
                        "partial",
                        MemoryStatus::Active,
                        "rust runtime",
                        1,
                    ),
                },
            ],
        ))
        .await
        .unwrap();

    let before = repository.snapshot().await;
    let first = repository
        .query(MemoryQuery::new(ns.clone()).with_text("rust memory"))
        .await
        .unwrap();
    let second = repository
        .query(MemoryQuery::new(ns).with_text("rust memory"))
        .await
        .unwrap();
    let after = repository.snapshot().await;

    assert_eq!(first, second);
    assert_eq!(before, after);
    assert_eq!(first.hits[0].node.id, "exact");
    assert!(first.hits[0].score.lexical > first.hits[1].score.lexical);
}

#[tokio::test]
async fn admission_and_use_events_are_independently_idempotent() {
    let repository = InMemoryRepository::new();
    let ns = namespace("repo-a");
    repository
        .apply(changes(
            ns.clone(),
            "seed",
            1,
            vec![MemoryOperation::Create {
                node: draft(ns.clone(), "one", MemoryStatus::Active, "memory", 1),
            }],
        ))
        .await
        .unwrap();

    let event =
        MemoryAccessEvent::new("event-1", ns.clone(), "one", 1, time(2)).with_context_id("run-1");
    repository.record_admission(event.clone()).await.unwrap();
    repository.record_admission(event.clone()).await.unwrap();
    repository.record_use(event.clone()).await.unwrap();
    repository.record_use(event).await.unwrap();

    let summary = repository.usage_summary(&ns, "one").await.unwrap();
    assert_eq!(summary.admissions, 1);
    assert_eq!(summary.uses, 1);

    let missing_revision = MemoryAccessEvent::new("event-2", ns.clone(), "one", 99, time(3));
    assert!(matches!(
        repository
            .record_admission(missing_revision)
            .await
            .unwrap_err(),
        MemoryRepositoryError::NodeRevisionNotFound { revision: 99, .. }
    ));

    let conflict = MemoryAccessEvent::new("event-1", ns.clone(), "one", 1, time(3));
    assert!(matches!(
        repository.record_use(conflict).await.unwrap_err(),
        MemoryRepositoryError::IdempotencyConflict { .. }
    ));
}

#[tokio::test]
async fn invalid_or_over_budget_changes_leave_state_unchanged() {
    let repository = InMemoryRepository::new();
    let ns = namespace("repo-a");
    let oversized = "x".repeat(MAX_CONTENT_BYTES + 1);
    let error = repository
        .apply(changes(
            ns.clone(),
            "oversized",
            1,
            vec![MemoryOperation::Create {
                node: draft(ns.clone(), "large", MemoryStatus::Active, &oversized, 1),
            }],
        ))
        .await
        .unwrap_err();
    assert!(matches!(error, MemoryRepositoryError::LimitExceeded { .. }));
    assert!(repository.snapshot().await.nodes.is_empty());
}

#[tokio::test]
async fn concurrent_writers_have_one_serializable_winner() {
    let repository = Arc::new(InMemoryRepository::new());
    let ns = namespace("repo-a");
    repository
        .apply(changes(
            ns.clone(),
            "seed",
            1,
            vec![MemoryOperation::Create {
                node: draft(ns.clone(), "one", MemoryStatus::Active, "before", 1),
            }],
        ))
        .await
        .unwrap();

    let update = |key: &str, content: &str, evidence_name: &str| {
        changes(
            ns.clone(),
            key,
            2,
            vec![MemoryOperation::Revise {
                node_id: "one".into(),
                expected_revision: 1,
                content: content.into(),
                mode: RevisionMode::Correction,
                evidence: vec![evidence(evidence_name, 2)],
                confidence: None,
                importance: None,
            }],
        )
    };

    let left_repo = repository.clone();
    let right_repo = repository.clone();
    let (left, right) = tokio::join!(
        left_repo.apply(update("left", "left wins", "left")),
        right_repo.apply(update("right", "right wins", "right")),
    );
    let loser = match (left, right) {
        (Err(error), Ok(_)) | (Ok(_), Err(error)) => error,
        (left, right) => panic!("expected one winner and one loser, got {left:?} and {right:?}"),
    };
    assert!(matches!(
        loser,
        MemoryRepositoryError::RevisionConflict { .. }
    ));

    let node = repository.get(&ns, "one").await.unwrap().unwrap();
    assert_eq!(node.revision, 2);
    assert!(matches!(node.content.as_str(), "left wins" | "right wins"));
}
