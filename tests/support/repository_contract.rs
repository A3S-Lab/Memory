use a3s_memory::repository::{
    DurableMemoryKind, EvidenceKind, EvidenceRef, MemoryAccessEvent, MemoryChangeSet,
    MemoryNamespace, MemoryNodeDraft, MemoryOperation, MemoryQuery, MemoryRepository, MemoryStatus,
    RevisionMode,
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
            }],
        ))
        .await
        .unwrap();

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
}
