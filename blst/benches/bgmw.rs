use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kzg::{Fr, G1Mul, G1};
use rust_kzg_blst::{
    msm::BGMWTable,
    types::{fr::FsFr, g1::FsG1},
};

fn bench_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multi-scalar multiplication");

    for i in 8..13usize {
        let point_count = 1usize << 12;

        let points = (0..point_count)
            .map(|_| FsG1::rand())
            .collect::<Vec<FsG1>>();
        let scalars = (0..point_count)
            .map(|_| FsFr::rand())
            .collect::<Vec<FsFr>>();

        let table = BGMWTable::compute(&points).unwrap();

        let input = (points, scalars);
        group.bench_with_input(
            BenchmarkId::new("Standard Pippenger's algorithm", i),
            &input,
            |b, (points, scalars)| {
                b.iter(|| FsG1::g1_lincomb(points, scalars, points.len(), None));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("BGMW algorithm", i),
            &input,
            |b, (points, scalars)| {
                b.iter(|| FsG1::g1_lincomb(points, scalars, points.len(), Some(&table)));
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(200);
    targets = bench_msm
}
criterion_main!(benches);
