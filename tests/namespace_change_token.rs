use a3s_memory::repository::{
    DurableMemoryKind, EvidenceKind, EvidenceRef, FileMemoryRepository, InMemoryRepository,
    MemoryAccessEvent, MemoryChangeResult, MemoryChangeSet, MemoryNamespace,
    MemoryNamespaceChangeToken, MemoryNode, MemoryNodeDraft, MemoryOperation, MemoryQuery,
    MemoryQueryResult, MemoryRepository, MemoryRepositoryError, MemoryStatus, MemoryUsageSummary,
    RevisionMode, MEMORY_NAMESPACE_CHANGE_TOKEN_PROFILE_V1,
};
use chrono::{TimeZone, Utc};
use std::sync::Arc;

fn time(second: u32) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 30, 15, 0, second)
        .single()
        .expect("valid test time")
}

fn namespace(scope: &str) -> MemoryNamespace {
    MemoryNamespace::try_new("token-tenant", "token-principal", scope).unwrap()
}

fn evidence(name: &str, second: u32) -> EvidenceRef {
    EvidenceRef::try_new(
        format!("a3s://token/run/turn/{name}"),
        format!("sha256:{name:0>64}"),
        EvidenceKind::SessionTurn,
        time(second),
    )
    .unwrap()
}

fn create_change(namespace: &MemoryNamespace, key: &str, node_id: &str) -> MemoryChangeSet {
    MemoryChangeSet::new(
        key,
        namespace.clone(),
        time(1),
        vec![MemoryOperation::Create {
            node: MemoryNodeDraft::new(
                node_id,
                namespace.clone(),
                DurableMemoryKind::Semantic,
                MemoryStatus::Active,
                "token contract content",
                vec![evidence(key, 1)],
                time(1),
            ),
        }],
    )
}

async fn token(
    repository: &dyn MemoryRepository,
    namespace: &MemoryNamespace,
) -> MemoryNamespaceChangeToken {
    repository
        .namespace_change_token(namespace)
        .await
        .unwrap()
        .expect("built-in repository change token")
}

fn assert_send_sync<T: Send + Sync>() {}

#[tokio::test]
async fn in_memory_tokens_linearize_novel_changes_without_counting_replays_or_access() {
    assert_send_sync::<MemoryNamespaceChangeToken>();
    let repository = Arc::new(InMemoryRepository::new());
    let primary = namespace("primary");
    let other = namespace("other");

    let initial = token(repository.as_ref(), &primary).await;
    initial.verify().unwrap();
    assert_eq!(initial.profile(), MEMORY_NAMESPACE_CHANGE_TOKEN_PROFILE_V1);
    assert_eq!(initial.sequence(), 0);
    assert_eq!(token(repository.as_ref(), &other).await.sequence(), 0);

    let create = create_change(&primary, "create-primary", "node");
    let first = repository.apply(create.clone()).await.unwrap();
    assert_eq!(token(repository.as_ref(), &primary).await.sequence(), 1);
    assert_eq!(repository.apply(create).await.unwrap(), first);
    assert_eq!(token(repository.as_ref(), &primary).await.sequence(), 1);

    repository
        .record_admission(MemoryAccessEvent::new(
            "admit-node",
            primary.clone(),
            "node",
            1,
            time(2),
        ))
        .await
        .unwrap();
    repository
        .record_use(MemoryAccessEvent::new(
            "use-node",
            primary.clone(),
            "node",
            1,
            time(3),
        ))
        .await
        .unwrap();
    assert_eq!(token(repository.as_ref(), &primary).await.sequence(), 1);

    let failed = MemoryChangeSet::new(
        "failed-revision",
        primary.clone(),
        time(4),
        vec![MemoryOperation::Revise {
            node_id: "node".into(),
            expected_revision: 99,
            content: "must not publish".into(),
            mode: RevisionMode::Correction,
            evidence: vec![evidence("failed", 4)],
            confidence: None,
            importance: None,
        }],
    );
    assert!(matches!(
        repository.apply(failed).await,
        Err(MemoryRepositoryError::RevisionConflict { .. })
    ));
    assert_eq!(token(repository.as_ref(), &primary).await.sequence(), 1);

    repository
        .apply(create_change(&other, "create-other", "other-node"))
        .await
        .unwrap();
    assert_eq!(token(repository.as_ref(), &primary).await.sequence(), 1);
    assert_eq!(token(repository.as_ref(), &other).await.sequence(), 1);

    let revise = |key: &str, content: &str| {
        MemoryChangeSet::new(
            key,
            primary.clone(),
            time(5),
            vec![MemoryOperation::Revise {
                node_id: "node".into(),
                expected_revision: 1,
                content: content.into(),
                mode: RevisionMode::Correction,
                evidence: vec![evidence(key, 5)],
                confidence: None,
                importance: None,
            }],
        )
    };
    let left = repository.clone();
    let right = repository.clone();
    let (left_result, right_result) = tokio::join!(
        left.apply(revise("left", "left wins")),
        right.apply(revise("right", "right wins")),
    );
    assert!(matches!(
        (&left_result, &right_result),
        (Ok(_), Err(MemoryRepositoryError::RevisionConflict { .. }))
            | (Err(MemoryRepositoryError::RevisionConflict { .. }), Ok(_))
    ));
    assert_eq!(token(repository.as_ref(), &primary).await.sequence(), 2);

    let encoded = serde_json::to_string(&token(repository.as_ref(), &primary).await).unwrap();
    assert!(!encoded.contains("token-tenant"));
    assert!(!encoded.contains("token-principal"));
    assert!(!encoded.contains("token contract content"));
    let mut tampered = serde_json::to_value(MemoryNamespaceChangeToken::new(2)).unwrap();
    tampered["profile"] = serde_json::json!("unsupported");
    let tampered: MemoryNamespaceChangeToken = serde_json::from_value(tampered).unwrap();
    assert!(tampered.verify().is_err());
}

#[tokio::test]
async fn file_tokens_reconstruct_exactly_across_restart() {
    let directory = tempfile::tempdir().unwrap();
    let namespace = namespace("file");
    let create = create_change(&namespace, "file-create", "node");
    {
        let repository = FileMemoryRepository::open(directory.path()).await.unwrap();
        repository.apply(create.clone()).await.unwrap();
        repository
            .apply(MemoryChangeSet::new(
                "file-revise",
                namespace.clone(),
                time(2),
                vec![MemoryOperation::Revise {
                    node_id: "node".into(),
                    expected_revision: 1,
                    content: "restarted content".into(),
                    mode: RevisionMode::Correction,
                    evidence: vec![evidence("file-revise", 2)],
                    confidence: None,
                    importance: None,
                }],
            ))
            .await
            .unwrap();
        assert_eq!(token(&repository, &namespace).await.sequence(), 2);
    }

    let reopened = FileMemoryRepository::open(directory.path()).await.unwrap();
    assert_eq!(token(&reopened, &namespace).await.sequence(), 2);
    reopened.apply(create).await.unwrap();
    assert_eq!(token(&reopened, &namespace).await.sequence(), 2);
}

struct UntrackedRepository {
    inner: InMemoryRepository,
}

#[async_trait::async_trait]
impl MemoryRepository for UntrackedRepository {
    async fn apply(
        &self,
        change_set: MemoryChangeSet,
    ) -> Result<MemoryChangeResult, MemoryRepositoryError> {
        self.inner.apply(change_set).await
    }

    async fn get(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<Option<MemoryNode>, MemoryRepositoryError> {
        self.inner.get(namespace, node_id).await
    }

    async fn query(&self, query: MemoryQuery) -> Result<MemoryQueryResult, MemoryRepositoryError> {
        self.inner.query(query).await
    }

    async fn record_admission(
        &self,
        event: MemoryAccessEvent,
    ) -> Result<(), MemoryRepositoryError> {
        self.inner.record_admission(event).await
    }

    async fn record_use(&self, event: MemoryAccessEvent) -> Result<(), MemoryRepositoryError> {
        self.inner.record_use(event).await
    }

    async fn usage_summary(
        &self,
        namespace: &MemoryNamespace,
        node_id: &str,
    ) -> Result<MemoryUsageSummary, MemoryRepositoryError> {
        self.inner.usage_summary(namespace, node_id).await
    }
}

#[tokio::test]
async fn custom_repositories_explicitly_opt_in_to_change_tokens() {
    let namespace = namespace("custom");
    let repository = UntrackedRepository {
        inner: InMemoryRepository::new(),
    };
    repository
        .apply(create_change(&namespace, "custom-create", "node"))
        .await
        .unwrap();
    assert!(repository
        .namespace_change_token(&namespace)
        .await
        .unwrap()
        .is_none());
}
