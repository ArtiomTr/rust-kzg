use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use kzg::eip_4844::Blob;
use kzg_bench::tests::eip_4844::generate_random_blob_bytes;
use rust_kzg_sppark::{blob_to_kzg_commitment, load_trusted_setup_file};

fn bench_eip_4844_(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let blob_bytes = generate_random_blob_bytes(&mut rng);
    let blob = Blob {
        bytes: blob_bytes
    };
    let settings = load_trusted_setup_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kzg-bench/src/trusted_setup.txt"));

    c.bench_function("blob_to_kzg_commitment", |b| {
        b.iter(|| blob_to_kzg_commitment(&blob, &settings))
    });
}

criterion_group!(benches, bench_eip_4844_);
criterion_main!(benches);
