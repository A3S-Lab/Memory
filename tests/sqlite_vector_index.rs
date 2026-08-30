#![cfg(feature = "sqlite")]

use a3s_memory::vector::{
    InMemoryVectorIndex, SqliteVectorIndex, VectorIndex, VectorIndexDescriptor, VectorIndexError,
    VectorMutationConsistency, VectorRecord, VectorRevision, VectorSearchRequest,
};
use rusqlite::{params, Connection};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Barrier;

fn record(id: &str, embedding: [f32; 2]) -> VectorRecord {
    VectorRecord::new(id, embedding.to_vec()).with_label("kind", "test")
}

fn database_path(directory: &TempDir) -> std::path::PathBuf {
    directory.path().join("vectors.sqlite3")
}

#[tokio::test]
async fn reopen_preserves_records_revision_and_exact_history_token() {
    let directory = TempDir::new().unwrap();
    let path = database_path(&directory);
    let descriptor = VectorIndexDescriptor::new(2);
    let initial;
    let published;

    {
        let index = SqliteVectorIndex::open(&path, descriptor.clone())
            .await
            .unwrap();
        assert_eq!(
            index.mutation_consistency(),
            VectorMutationConsistency::IndexRevisionCas
        );
        initial = index.observe().await.unwrap();
        published = index
            .replace_partition_if_revision(
                "semantic",
                initial.status.revision,
                vec![record("alpha", [2.0, 0.0]), record("beta", [0.0, 3.0])],
            )
            .await
            .unwrap();
        assert_eq!(published.revision, VectorRevision::new(1));
    }

    let reopened = SqliteVectorIndex::open(&path, descriptor).await.unwrap();
    let observation = reopened.observe().await.unwrap();
    assert_eq!(observation.status, published);
    assert_eq!(
        observation.change_token.as_ref().unwrap().history_digest(),
        initial.change_token.as_ref().unwrap().history_digest()
    );
    assert_eq!(
        observation.change_token.as_ref().unwrap().revision(),
        VectorRevision::new(1)
    );

    let result = reopened
        .search(
            VectorSearchRequest::new(vec![1.0, 0.0], 10)
                .with_partition("semantic")
                .with_label("kind", "test"),
        )
        .await
        .unwrap();
    assert_eq!(result.status, observation.status);
    assert_eq!(result.hits.len(), 2);
    assert_eq!(result.hits[0].id, "alpha");
}

#[tokio::test]
async fn independent_connections_serialize_global_revision_cas() {
    let directory = TempDir::new().unwrap();
    let path = database_path(&directory);
    let descriptor = VectorIndexDescriptor::new(2);
    let left = Arc::new(
        SqliteVectorIndex::open(&path, descriptor.clone())
            .await
            .unwrap(),
    );
    let right = Arc::new(SqliteVectorIndex::open(&path, descriptor).await.unwrap());
    let expected = left.observe().await.unwrap().status.revision;
    let barrier = Arc::new(Barrier::new(3));
    let mut tasks = Vec::new();

    for (index, id, embedding) in [
        (Arc::clone(&left), "left", [1.0, 0.0]),
        (Arc::clone(&right), "right", [0.0, 1.0]),
    ] {
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            index
                .replace_partition_if_revision("semantic", expected, vec![record(id, embedding)])
                .await
        }));
    }
    barrier.wait().await;

    let mut successes = 0;
    let mut conflicts = 0;
    for task in tasks {
        match task.await.unwrap() {
            Ok(status) => {
                successes += 1;
                assert_eq!(status.revision, VectorRevision::new(1));
            }
            Err(VectorIndexError::RevisionConflict { expected, actual }) => {
                conflicts += 1;
                assert_eq!(expected, VectorRevision::new(0));
                assert_eq!(actual, VectorRevision::new(1));
            }
            Err(error) => panic!("unexpected durable CAS result: {error}"),
        }
    }

    assert_eq!(successes, 1);
    assert_eq!(conflicts, 1);
    assert_eq!(
        left.observe().await.unwrap(),
        right.observe().await.unwrap()
    );
}

#[tokio::test]
async fn descriptor_drift_fails_closed_without_reinitializing_storage() {
    let directory = TempDir::new().unwrap();
    let path = database_path(&directory);
    let original = SqliteVectorIndex::open(&path, VectorIndexDescriptor::new(2))
        .await
        .unwrap();
    original
        .replace_partition("semantic", vec![record("kept", [1.0, 0.0])])
        .await
        .unwrap();
    let before = original.observe().await.unwrap();
    drop(original);

    let error = SqliteVectorIndex::open(&path, VectorIndexDescriptor::new(3))
        .await
        .unwrap_err();
    assert_eq!(error, VectorIndexError::DescriptorMismatch);

    let reopened = SqliteVectorIndex::open(&path, VectorIndexDescriptor::new(2))
        .await
        .unwrap();
    assert_eq!(reopened.observe().await.unwrap(), before);
}

#[tokio::test]
async fn corrupted_metadata_fails_closed_on_reopen() {
    let directory = TempDir::new().unwrap();
    let path = database_path(&directory);
    let index = SqliteVectorIndex::open(&path, VectorIndexDescriptor::new(2))
        .await
        .unwrap();
    index
        .replace_partition("semantic", vec![record("alpha", [1.0, 0.0])])
        .await
        .unwrap();
    drop(index);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE a3s_vector_index_metadata SET record_count = ?1 WHERE singleton = 1",
            params![99_i64],
        )
        .unwrap();
    drop(connection);

    let error = SqliteVectorIndex::open(&path, VectorIndexDescriptor::new(2))
        .await
        .unwrap_err();
    assert!(matches!(error, VectorIndexError::StorageCorrupted(_)));
}

#[tokio::test]
async fn corrupted_vector_content_fails_closed_even_when_accounting_still_matches() {
    let directory = TempDir::new().unwrap();
    let path = database_path(&directory);
    let index = SqliteVectorIndex::open(&path, VectorIndexDescriptor::new(2))
        .await
        .unwrap();
    index
        .replace_partition("semantic", vec![record("alpha", [1.0, 0.0])])
        .await
        .unwrap();
    drop(index);

    let connection = Connection::open(&path).unwrap();
    let replacement = [0.0_f32, 1.0_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    connection
        .execute(
            "UPDATE a3s_vector_records SET embedding = ?1
             WHERE partition = 'semantic' AND position = 0",
            params![replacement],
        )
        .unwrap();
    drop(connection);

    let error = SqliteVectorIndex::open(&path, VectorIndexDescriptor::new(2))
        .await
        .unwrap_err();
    assert!(matches!(error, VectorIndexError::StorageCorrupted(_)));
}

#[tokio::test]
async fn no_op_mutations_preserve_revision_and_token_across_reopen() {
    let directory = TempDir::new().unwrap();
    let path = database_path(&directory);
    let descriptor = VectorIndexDescriptor::new(2);
    let index = SqliteVectorIndex::open(&path, descriptor.clone())
        .await
        .unwrap();
    let initial = index.observe().await.unwrap();

    assert_eq!(
        index
            .replace_partition("missing", Vec::new())
            .await
            .unwrap(),
        initial.status
    );
    assert_eq!(
        index.remove_partition("missing").await.unwrap(),
        initial.status
    );
    assert_eq!(index.clear().await.unwrap(), initial.status);
    assert_eq!(index.observe().await.unwrap(), initial);
    drop(index);

    let reopened = SqliteVectorIndex::open(&path, descriptor).await.unwrap();
    assert_eq!(reopened.observe().await.unwrap(), initial);
}

#[tokio::test]
async fn durable_and_in_memory_indexes_share_stable_logical_byte_accounting() {
    let directory = TempDir::new().unwrap();
    let path = database_path(&directory);
    let descriptor = VectorIndexDescriptor::new(2);
    let memory = InMemoryVectorIndex::new(descriptor.clone()).unwrap();
    let durable = SqliteVectorIndex::open(&path, descriptor).await.unwrap();
    let records = vec![
        record("alpha", [1.0, 0.0]),
        record("beta", [0.0, 1.0]).with_label("scope", "workspace"),
    ];

    let memory_status = memory
        .replace_partition("semantic", records.clone())
        .await
        .unwrap();
    let durable_status = durable
        .replace_partition("semantic", records)
        .await
        .unwrap();

    assert_eq!(durable_status.byte_count, memory_status.byte_count);
    assert_eq!(durable_status.record_count, memory_status.record_count);
    assert_eq!(
        durable_status.partition_count,
        memory_status.partition_count
    );
}
