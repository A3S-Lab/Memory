#![cfg(feature = "sqlite-vec")]

use a3s_memory::SqliteMemoryStore;

#[tokio::test]
async fn concurrent_first_opens_register_vec_before_creating_connections() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();

    let (left_store, right_store) = tokio::join!(
        SqliteMemoryStore::new(left.path()),
        SqliteMemoryStore::new(right.path()),
    );

    left_store.expect("first SQLite connection must load vec0");
    right_store.expect("concurrent SQLite connection must load vec0");
}
