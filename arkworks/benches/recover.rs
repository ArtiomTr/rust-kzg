use criterion::{criterion_group, criterion_main, Criterion};
use kzg_bench::benches::recover::bench_recover;

use rust_kzg_arkworks::kzg_proofs::LFFTSettings;
use rust_kzg_arkworks::kzg_types::ArkFr;
use rust_kzg_arkworks::utils::PolyData;

fn bench_recover_(c: &mut Criterion) {
    bench_recover::<ArkFr, LFFTSettings, PolyData, PolyData>(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_recover_
}

criterion_main!(benches);
