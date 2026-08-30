use a3s_memory::vector::{
    InMemoryVectorIndex, VectorIndex, VectorIndexChangeToken, VectorIndexDescriptor, VectorRecord,
    VectorRevision, VECTOR_INDEX_CHANGE_TOKEN_PROFILE_V1,
};

fn record(id: &str, embedding: [f32; 2]) -> VectorRecord {
    VectorRecord::new(id, embedding.to_vec())
}

#[tokio::test]
async fn in_memory_change_tokens_bind_one_index_history_and_exact_revision() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VectorIndexChangeToken>();

    let first = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    let first_clone = first.clone();
    let initial = first.change_token().expect("built-in change token");
    initial.verify().unwrap();
    assert_eq!(initial.profile(), VECTOR_INDEX_CHANGE_TOKEN_PROFILE_V1);
    assert_eq!(initial.revision(), VectorRevision::new(0));
    assert_eq!(first_clone.change_token(), Some(initial.clone()));

    let encoded = serde_json::to_string(&initial).unwrap();
    let decoded: VectorIndexChangeToken = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, initial);
    let mut unknown = serde_json::to_value(&initial).unwrap();
    unknown["stateDigest"] = serde_json::json!("forged");
    assert!(serde_json::from_value::<VectorIndexChangeToken>(unknown).is_err());

    first
        .replace_partition("active", vec![record("alpha", [1.0, 0.0])])
        .await
        .unwrap();
    let published = first.change_token().expect("published token");
    assert_eq!(published.history_id(), initial.history_id());
    assert_eq!(published.revision(), VectorRevision::new(1));
    assert_ne!(published, initial);

    first.remove_partition("missing").await.unwrap();
    assert_eq!(
        first.change_token(),
        Some(published.clone()),
        "a no-op mutation must not advance the token"
    );

    let unrelated = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    unrelated
        .replace_partition("active", vec![record("bravo", [0.0, 1.0])])
        .await
        .unwrap();
    assert_eq!(unrelated.status(), first.status());
    let unrelated_token = unrelated.change_token().expect("unrelated token");
    assert_eq!(unrelated_token.revision(), published.revision());
    assert_ne!(unrelated_token.history_id(), published.history_id());
    assert_ne!(unrelated_token, published);
}

#[test]
fn custom_change_tokens_validate_bounded_history_identity_and_profile() {
    let token = VectorIndexChangeToken::try_new("remote-cluster.index-17", VectorRevision::new(9))
        .expect("valid custom history");
    token.verify().unwrap();

    assert!(VectorIndexChangeToken::try_new("", VectorRevision::new(1)).is_err());
    assert!(VectorIndexChangeToken::try_new(" history", VectorRevision::new(1)).is_err());
    assert!(VectorIndexChangeToken::try_new("history/unsafe", VectorRevision::new(1)).is_err());

    let mut forged = serde_json::to_value(token).unwrap();
    forged["profile"] = serde_json::json!("a3s.memory.vector-index-change-token.v2");
    let forged: VectorIndexChangeToken = serde_json::from_value(forged).unwrap();
    assert!(forged.verify().is_err());
}
