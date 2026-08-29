use a3s_memory::repository::{
    DurableMemoryKind, EvidenceKind, EvidenceRef, MemoryAccessEvent, MemoryChangeSet,
    MemoryNamespace, MemoryNodeDraft, MemoryOperation, MemoryQuery, MemoryRepository,
    MemoryRepositoryError, MemorySnapshotRequest, MemoryStatus, RevisionMode,
    MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1,
};
use chrono::{TimeZone, Utc};

fn time(second: u32) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 29, 13, 0, second)
        .single()
        .expect("valid contract time")
}

fn evidence(name: &str, second: u32) -> EvidenceRef {
    EvidenceRef::try_new(
        format!("a3s://contract/run/turn/{name}"),
        format!("sha256:{name:0>64}"),
        EvidenceKind::SessionTurn,
        time(second),
    )
    .expect("valid contract evidence")
}

pub async fn assert_repository_contract(repository: &dyn MemoryRepository, scope: &str) {
    let namespace = MemoryNamespace::try_new("contract-tenant", "contract-user", scope).unwrap();
    let other = MemoryNamespace::try_new("contract-tenant", "contract-user", "other").unwrap();
    let create = MemoryChangeSet::new(
        "contract-create",
        namespace.clone(),
        time(1),
        vec![MemoryOperation::Create {
            node: MemoryNodeDraft::new(
                "contract-node",
                namespace.clone(),
                DurableMemoryKind::Semantic,
                MemoryStatus::Candidate,
                "remember the stable contract",
                vec![evidence("create", 1)],
                time(1),
            ),
        }],
    );

    let first = repository.apply(create.clone()).await.unwrap();
    assert_eq!(repository.apply(create).await.unwrap(), first);
    assert!(repository
        .get(&other, "contract-node")
        .await
        .unwrap()
        .is_none());
    assert!(repository
        .query(MemoryQuery::new(namespace.clone()))
        .await
        .unwrap()
        .hits
        .is_empty());

    repository
        .apply(MemoryChangeSet::new(
            "contract-activate",
            namespace.clone(),
            time(2),
            vec![MemoryOperation::Activate {
                node_id: "contract-node".into(),
                expected_revision: 1,
                evidence: vec![EvidenceRef::try_new(
                    "a3s://contract/review/activation",
                    format!("sha256:{:0>64}", "activation"),
                    EvidenceKind::Verification,
                    time(2),
                )
                .unwrap()],
            }],
        ))
        .await
        .unwrap();

    let active_snapshot = repository
        .snapshot_namespace(MemorySnapshotRequest::new(namespace.clone(), 4))
        .await
        .unwrap();
    assert_eq!(
        active_snapshot.profile,
        MEMORY_NAMESPACE_SNAPSHOT_PROFILE_V1
    );
    assert_eq!(active_snapshot.namespace, namespace);
    assert_eq!(active_snapshot.statuses, [MemoryStatus::Active].into());
    assert_eq!(active_snapshot.nodes.len(), 1);
    assert_eq!(active_snapshot.nodes[0].id, "contract-node");
    assert_eq!(active_snapshot.nodes[0].revision, 2);
    assert!(active_snapshot.digest.starts_with("sha256:"));
    assert_eq!(
        repository
            .snapshot_namespace(MemorySnapshotRequest::new(namespace.clone(), 4))
            .await
            .unwrap(),
        active_snapshot,
        "an unchanged exact namespace view must have a stable snapshot identity"
    );

    let first_query = repository
        .query(MemoryQuery::new(namespace.clone()).with_text("stable contract"))
        .await
        .unwrap();
    let second_query = repository
        .query(MemoryQuery::new(namespace.clone()).with_text("stable contract"))
        .await
        .unwrap();
    assert_eq!(first_query, second_query);
    assert_eq!(first_query.hits.len(), 1);
    assert_eq!(first_query.hits[0].node.revision, 2);

    let admission = MemoryAccessEvent::new(
        "contract-admission",
        namespace.clone(),
        "contract-node",
        2,
        time(3),
    );
    repository
        .record_admission(admission.clone())
        .await
        .unwrap();
    repository.record_admission(admission).await.unwrap();
    assert_eq!(
        repository
            .usage_summary(&namespace, "contract-node")
            .await
            .unwrap()
            .admissions,
        1
    );

    repository
        .apply(MemoryChangeSet::new(
            "contract-revise",
            namespace.clone(),
            time(4),
            vec![MemoryOperation::Revise {
                node_id: "contract-node".into(),
                expected_revision: 2,
                content: "remember the corrected contract".into(),
                mode: RevisionMode::Correction,
                evidence: vec![evidence("correction", 4)],
                confidence: Some(0.9),
                importance: None,
            }],
        ))
        .await
        .unwrap();
    let corrected = repository
        .get(&namespace, "contract-node")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(corrected.revision, 3);
    assert_eq!(corrected.history.len(), 2);
    assert_eq!(corrected.history[0].content, "remember the stable contract");

    repository
        .apply(MemoryChangeSet::new(
            "contract-create-cjk",
            namespace.clone(),
            time(5),
            vec![MemoryOperation::Create {
                node: MemoryNodeDraft::new(
                    "contract-cjk-node",
                    namespace.clone(),
                    DurableMemoryKind::Procedural,
                    MemoryStatus::Candidate,
                    "部署前执行数据库迁移并验证架构版本",
                    vec![evidence("cjk-create", 5)],
                    time(5),
                ),
            }],
        ))
        .await
        .unwrap();
    repository
        .apply(MemoryChangeSet::new(
            "contract-activate-cjk",
            namespace.clone(),
            time(6),
            vec![MemoryOperation::Activate {
                node_id: "contract-cjk-node".into(),
                expected_revision: 1,
                evidence: vec![EvidenceRef::try_new(
                    "a3s://contract/review/cjk-activation",
                    format!("sha256:{:0>64}", "cjk-activation"),
                    EvidenceKind::Verification,
                    time(6),
                )
                .unwrap()],
            }],
        ))
        .await
        .unwrap();

    let cjk_query = repository
        .query(MemoryQuery::new(namespace.clone()).with_text("数据库迁移验证"))
        .await
        .unwrap();
    assert_eq!(cjk_query.hits.len(), 1);
    assert_eq!(cjk_query.hits[0].node.id, "contract-cjk-node");
    assert!(cjk_query.hits[0].score.lexical > 0.5);

    let current = repository
        .snapshot_namespace(MemorySnapshotRequest::new(namespace.clone(), 2))
        .await
        .unwrap();
    assert_eq!(
        current
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["contract-cjk-node", "contract-node"],
        "snapshot ordering must be backend-independent"
    );
    assert_ne!(current.digest, active_snapshot.digest);

    let overflow = repository
        .snapshot_namespace(MemorySnapshotRequest::new(namespace, 1))
        .await
        .unwrap_err();
    assert!(matches!(
        overflow,
        MemoryRepositoryError::LimitExceeded {
            resource,
            limit: 1,
            actual: 2,
        } if resource == "namespace snapshot nodes"
    ));
}
