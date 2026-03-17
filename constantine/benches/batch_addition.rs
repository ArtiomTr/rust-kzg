use criterion::{criterion_group, criterion_main, Criterion};
use kzg_bench::benches::batch_addition::bench_batch_addition;
use rust_kzg_constantine::eip_7594::CtBackend;

fn bench_batch_addition_(c: &mut Criterion) {
    bench_batch_addition::<CtBackend>(c)
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_batch_addition_
}

criterion_main!(benches);
