use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use blst::{blst_p1s_mult_wbits, blst_p1s_mult_wbits_precompute};
use human_bytes::human_bytes;
use kzg::{msm::precompute::precompute, Fr, G1Affine, G1};
use rust_kzg_blst::types::{fp::FsFp, fr::FsFr, g1::{FsG1, FsG1Affine}};
use crate_crypto_internal_eth_kzg_bls12_381::{ff::Field, fixed_base_msm::FixedBaseMSM, group::Group};

fn bench_fixed_base_msm(c: &mut Criterion) {
    let npow = 12usize;
    let npoints = 1usize << npow;

    {
        let points = (0..npoints).map(|_| {
            FsG1Affine::into_affine(&FsG1::rand()).0
        }).collect::<Vec<_>>();
        let points = [points.as_ptr(), std::ptr::null()];
        let mut group = c.benchmark_group("blst wbits initialization");
        let precomputations = (8..=10).map(|wbits| {
            let precompute_size = unsafe { blst::blst_p1s_mult_wbits_precompute_sizeof(wbits, npoints) };
            let mut precomputation = vec![blst::blst_p1_affine::default(); precompute_size / size_of::<blst::blst_p1_affine>()];

            group.bench_function(BenchmarkId::from_parameter(format!("points: 2^{}, wbits: {}, precomp_size: {}", npow, wbits, human_bytes(precompute_size as f64))), |b| {
                b.iter(|| {
                    unsafe { blst_p1s_mult_wbits_precompute(precomputation.as_mut_ptr(), wbits, points.as_ptr(), npoints) };
                });
            });

            (wbits, precomputation)
        }).collect::<Vec<_>>();
        group.finish();

        let scalars = (0..npoints).map(|_| {
            let fr = FsFr::rand().0;
            let mut scalar = blst::blst_scalar::default();
            unsafe { blst::blst_scalar_from_fr(&mut scalar, &fr); }
            scalar.b
        }).collect::<Vec<_>>();

        let mut group = c.benchmark_group("blst wbits mult");
        precomputations.into_iter().for_each(|(wbits, precomputation)| {
            let scratch_size = unsafe { blst::blst_p1s_mult_wbits_scratch_sizeof(npoints) };
            let mut scratch = vec![blst::limb_t::default(); scratch_size / size_of::<blst::limb_t>()];
            let scalars_arg = [scalars.as_ptr() as *const u8, std::ptr::null()];
            
            group.bench_function(BenchmarkId::from_parameter(format!("points: 2^{}, wbits: {}", npow, wbits)), |b| {
                b.iter(|| {
                    let mut output = blst::blst_p1::default();
                    unsafe { blst_p1s_mult_wbits(&mut output, precomputation.as_ptr(), wbits, npoints, scalars_arg.as_ptr(), 255, scratch.as_mut_ptr()); }
                })
            });
        });
        group.finish();
    }

    {
        let mut rng = rand::thread_rng();
        let points = (0..npoints).map(|_| {
            blstrs::G1Projective::random(&mut rng).into()
        }).collect::<Vec<_>>();
        let mut group = c.benchmark_group("crate-crypto wbits initialization");
        let precomputations = (8..=10).map(|wbits| {
            group.bench_function(BenchmarkId::from_parameter(format!("points: 2^{}, wbits: {}", npow, wbits)), |b| {
                b.iter(|| {
                    FixedBaseMSM::new(points.clone(), crate_crypto_internal_eth_kzg_bls12_381::fixed_base_msm::UsePrecomp::Yes { width: wbits })
                });
            });

            (wbits, FixedBaseMSM::new(points.clone(), crate_crypto_internal_eth_kzg_bls12_381::fixed_base_msm::UsePrecomp::Yes { width: wbits }))
        }).collect::<Vec<_>>();
        group.finish();

        let scalars = (0..npoints).map(|_| {
            blstrs::Scalar::random(&mut rng)
        }).collect::<Vec<_>>();

        let mut group = c.benchmark_group("crate-crypto wbits mult");
        precomputations.into_iter().for_each(|(wbits, precomputation)| {
            group.bench_function(BenchmarkId::from_parameter(format!("points: 2^{}, wbits: {}", npow, wbits)), |b| {
                b.iter(|| {
                    precomputation.msm(scalars.clone())
                })
            });
        });
        group.finish();
    }

    {
        let points = (0..npoints).map(|_| {
            FsG1::rand()
        }).collect::<Vec<_>>();
        c.bench_function(format!("rust-kzg-blst msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                precompute::<FsFr, FsG1, FsFp, FsG1Affine>(&points).unwrap().unwrap()
            });
        });

        let table = precompute::<FsFr, FsG1, FsFp, FsG1Affine>(&points).unwrap().unwrap();

        let scalars = (0..npoints).map(|_| {
            FsFr::default()
        }).collect::<Vec<_>>();

        c.bench_function(format!("rust-kzg-blst msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                let _ = table.multiply_sequential(&scalars);
            })
        });
    }

    {
        let points = (0..npoints).map(|_| {
            FsG1::rand()
        }).collect::<Vec<_>>();
        c.bench_function(format!("rust-kzg-blst msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                precompute::<FsFr, FsG1, FsFp, FsG1Affine>(&points).unwrap().unwrap()
            });
        });

        let table = precompute::<FsFr, FsG1, FsFp, FsG1Affine>(&points).unwrap().unwrap();

        let scalars = (0..npoints).map(|_| {
            FsFr::default()
        }).collect::<Vec<_>>();

        c.bench_function(format!("rust-kzg-blst msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                let _ = table.multiply_sequential(&scalars);
            })
        });
    }

    {
        use rust_kzg_arkworks4::kzg_types::{ArkG1, ArkFp, ArkFr, ArkG1Affine};
        let points = (0..npoints).map(|_| {
            ArkG1::rand()
        }).collect::<Vec<_>>();
        c.bench_function(format!("rust-kzg-arkworks4 msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                precompute::<ArkFr, ArkG1, ArkFp, ArkG1Affine>(&points).unwrap().unwrap()
            });
        });

        let table = precompute::<ArkFr, ArkG1, ArkFp, ArkG1Affine>(&points).unwrap().unwrap();

        let scalars = (0..npoints).map(|_| {
            ArkFr::default()
        }).collect::<Vec<_>>();

        c.bench_function(format!("rust-kzg-arkworks4 msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                let _ = table.multiply_sequential(&scalars);
            })
        });
    }

    {
        use rust_kzg_arkworks5::kzg_types::{ArkG1, ArkFp, ArkFr, ArkG1Affine};
        let points = (0..npoints).map(|_| {
            ArkG1::rand()
        }).collect::<Vec<_>>();
        c.bench_function(format!("rust-kzg-arkworks5 msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                precompute::<ArkFr, ArkG1, ArkFp, ArkG1Affine>(&points).unwrap().unwrap()
            });
        });

        let table = precompute::<ArkFr, ArkG1, ArkFp, ArkG1Affine>(&points).unwrap().unwrap();

        let scalars = (0..npoints).map(|_| {
            ArkFr::default()
        }).collect::<Vec<_>>();

        c.bench_function(format!("rust-kzg-arkworks5 msm initialization, points: 2^{}", npow).as_str(), |b| {
            b.iter(|| {
                let _ = table.multiply_sequential(&scalars);
            })
        });
    }
}

criterion_group!(benches, bench_fixed_base_msm);
criterion_main!(benches);
