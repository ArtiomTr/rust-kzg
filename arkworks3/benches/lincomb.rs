use criterion::{criterion_group, criterion_main, Criterion};
use kzg_bench::benches::lincomb::bench_g1_lincomb;
use rust_kzg_arkworks3::fft_g1::g1_linear_combination;
use rust_kzg_arkworks3::kzg_types::{ArkFp, ArkFr, ArkG1, ArkG1Affine, ArkG1ProjAddAffine};

fn bench_g1_lincomb_(c: &mut Criterion) {
    bench_g1_lincomb::<ArkFr, ArkG1, ArkFp, ArkG1Affine, ArkG1ProjAddAffine>(
        c,
        &g1_linear_combination,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_g1_lincomb_
}

criterion_main!(benches);
