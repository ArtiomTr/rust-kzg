use criterion::{criterion_group, criterion_main, Criterion};
use kzg_bench::benches::das::bench_das_extension;
use rust_kzg_arkworks::kzg_proofs::LFFTSettings;
use rust_kzg_arkworks::kzg_types::ArkFr;

fn bench_das_extension_(c: &mut Criterion) {
    bench_das_extension::<ArkFr, LFFTSettings>(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_das_extension_
}

criterion_main!(benches);
