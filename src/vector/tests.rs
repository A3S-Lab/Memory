use super::{InMemoryVectorIndex, VectorIndex, VectorIndexDescriptor};
#[test]
fn invalid_descriptors_fail_before_allocating_index_state() {
    assert!(InMemoryVectorIndex::new(VectorIndexDescriptor::new(0)).is_err());
    assert!(InMemoryVectorIndex::new(VectorIndexDescriptor::new(3).with_max_records(0)).is_err());
    assert!(InMemoryVectorIndex::new(VectorIndexDescriptor::new(3).with_max_bytes(0)).is_err());
}

#[tokio::test]
async fn clear_is_revisioned_once_and_idempotent_when_empty() {
    let index = InMemoryVectorIndex::new(VectorIndexDescriptor::new(2)).unwrap();
    index
        .replace_partition(
            "source",
            vec![super::VectorRecord::new("record", vec![1.0, 0.0])],
        )
        .await
        .unwrap();

    let cleared = index.clear().await.unwrap();
    assert_eq!(cleared.revision.value(), 2);
    assert_eq!(cleared.partition_count, 0);
    assert_eq!(cleared.record_count, 0);
    assert_eq!(cleared.byte_count, 0);

    let unchanged = index.clear().await.unwrap();
    assert_eq!(unchanged.revision.value(), 2);
}
