use criterion::{BatchSize, Criterion};
use kzg::{msm::batch_addition::multi_batch_addition_binary_tree_stride, EcBackend, G1Affine, G1};
use rand::{thread_rng, Rng};

pub fn bench_batch_addition<B: EcBackend>(c: &mut Criterion) {
    let batches = 32usize;
    let numpoints = 1usize << 14;

    let mut rng = thread_rng();

    let points = (0..batches)
        .map(|_| {
            let mut pts = (0..numpoints)
                .map(|_| B::G1Affine::into_affine(&B::G1::rand()))
                .collect::<Vec<_>>();

            for _ in 0..rng.gen_range(3..10) {
                pts[rng.gen_range(0..numpoints)] = B::G1Affine::zero();
            }

            pts
        })
        .collect::<Vec<_>>();

    c.bench_function("multi_batch_addition_binary_tree_stride", |b| {
        b.iter_batched(
            || points.clone(),
            |points| {
                multi_batch_addition_binary_tree_stride::<
                    B::G1,
                    B::G1Fp,
                    B::G1Affine,
                    B::G1ProjAddAffine,
                >(points)
            },
            BatchSize::SmallInput,
        );
    });
}
