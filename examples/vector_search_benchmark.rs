//! Locked release qualification for the ephemeral in-memory vector index.
//!
//! Run from the Memory repository root:
//!
//! `cargo run --example vector_search_benchmark --release`

use a3s_memory::{
    InMemoryVectorIndex, VectorIndex, VectorIndexDescriptor, VectorRecord, VectorSearchRequest,
};
use anyhow::{bail, Context, Result};
use serde_json::json;
use std::time::{Duration, Instant};

const RECORD_COUNT: usize = 25_000;
const DIMENSION: usize = 384;
const TOP_K: usize = 20;
const WARMUP_SAMPLES: usize = 20;
const MEASURED_SAMPLES: usize = 100;
const P95_BUDGET_MS: f64 = 30.0;
const MAX_VECTOR_BYTES: usize = 128 * 1024 * 1024;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        bail!("vector search qualification must run with --release");
    }

    let descriptor = VectorIndexDescriptor::new(DIMENSION)
        .with_max_records(RECORD_COUNT)
        .with_max_bytes(MAX_VECTOR_BYTES);
    let index = InMemoryVectorIndex::new(descriptor)?;
    let records = (0..RECORD_COUNT)
        .map(|index| VectorRecord::new(format!("record-{index:05}"), basis_vector(index)))
        .collect();
    let build_started = Instant::now();
    let status = index.replace_partition("qualification", records).await?;
    let build_ms = elapsed_ms(build_started.elapsed());
    if status.record_count != RECORD_COUNT {
        bail!(
            "index admitted {} records instead of {RECORD_COUNT}",
            status.record_count
        );
    }

    let query = basis_vector(17);
    for _ in 0..WARMUP_SAMPLES {
        verify_top_k(
            index
                .search(VectorSearchRequest::new(query.clone(), TOP_K))
                .await?,
        )?;
    }

    let mut durations = Vec::with_capacity(MEASURED_SAMPLES);
    for _ in 0..MEASURED_SAMPLES {
        let started = Instant::now();
        let result = index
            .search(VectorSearchRequest::new(query.clone(), TOP_K))
            .await?;
        durations.push(started.elapsed());
        verify_top_k(result)?;
    }
    durations.sort_unstable();

    let p50_ms = percentile_ms(&durations, 50);
    let p95_ms = percentile_ms(&durations, 95);
    let max_ms = elapsed_ms(*durations.last().context("no benchmark samples")?);
    let accounted_bytes = index.status().byte_count;
    index.clear().await?;
    if index.status().record_count != 0 || index.status().byte_count != 0 {
        bail!("vector index retained memory after clear");
    }

    let passed = p95_ms <= P95_BUDGET_MS;
    let report = json!({
        "schemaVersion": 1,
        "profile": "ephemeral-vector-search-v1",
        "build": "release",
        "machine": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "logicalCpus": std::thread::available_parallelism().map(|value| value.get()).unwrap_or(1),
            "processor": std::env::var("PROCESSOR_IDENTIFIER").ok(),
        },
        "parameters": {
            "records": RECORD_COUNT,
            "dimension": DIMENSION,
            "topK": TOP_K,
            "warmupSamples": WARMUP_SAMPLES,
            "measuredSamples": MEASURED_SAMPLES,
        },
        "measurement": {
            "p50Ms": p50_ms,
            "p95Ms": p95_ms,
            "maxMs": max_ms,
            "buildMs": build_ms,
            "accountedBytes": accounted_bytes,
            "budgetP95Ms": P95_BUDGET_MS,
            "passed": passed,
        },
        "passed": passed,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);

    if !passed {
        bail!("vector search qualification failed: p95 {p95_ms:.3} ms");
    }
    Ok(())
}

fn verify_top_k(result: a3s_memory::VectorSearchResult) -> Result<()> {
    if result.hits.len() != TOP_K || result.searched_records != RECORD_COUNT {
        bail!(
            "search returned {} hits after scanning {} records",
            result.hits.len(),
            result.searched_records
        );
    }
    if result.hits.iter().any(|hit| (hit.score - 1.0).abs() > 1e-6) {
        bail!("top-k contained a non-exact score");
    }
    Ok(())
}

fn basis_vector(seed: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMENSION];
    vector[seed % DIMENSION] = 1.0;
    vector
}

fn percentile_ms(durations: &[Duration], percentile: usize) -> f64 {
    let rank = (durations.len() * percentile)
        .div_ceil(100)
        .saturating_sub(1);
    elapsed_ms(durations[rank.min(durations.len() - 1)])
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
