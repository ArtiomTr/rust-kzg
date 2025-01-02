extern crate alloc;

use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::types::fp::FsFp;
use crate::types::g1::FsG1;
use crate::types::{fr::FsFr, g1::FsG1Affine};

use kzg::msm::cell::Cell;
use kzg::msm::precompute::PrecomputationTable;

use crate::types::g2::FsG2;
use blst::{
    blst_fp12_is_one, blst_p1_affine, blst_p1_cneg, blst_p1_to_affine, blst_p2_affine, blst_p2_to_affine, blst_scalar, Pairing
};

use kzg::{eth, PairingVerify};

impl PairingVerify<FsG1, FsG2> for FsG1 {
    fn verify(a1: &FsG1, a2: &FsG2, b1: &FsG1, b2: &FsG2) -> bool {
        pairings_verify(a1, a2, b1, b2)
    }
}

pub fn g1_linear_combination(
    out: &mut FsG1,
    points: &[FsG1],
    scalars: &[FsFr],
    len: usize,
    precomputation: Option<&PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>>,
) {
    #[cfg(feature = "sppark")]
    {
        use blst::blst_fr;
        use kzg::{G1Mul, G1};

        if len < 8 {
            *out = FsG1::default();
            for i in 0..len {
                let tmp = points[i].mul(&scalars[i]);
                out.add_or_dbl_assign(&tmp);
            }

            return;
        }

        let scalars =
            unsafe { alloc::slice::from_raw_parts(scalars.as_ptr() as *const blst_fr, len) };

        let point = if let Some(precomputation) = precomputation {
            rust_kzg_blst_sppark::multi_scalar_mult_prepared(precomputation.table, scalars)
        } else {
            let affines = kzg::msm::msm_impls::batch_convert::<FsG1, FsFp, FsG1Affine>(&points);
            let affines = unsafe {
                alloc::slice::from_raw_parts(affines.as_ptr() as *const blst_p1_affine, len)
            };
            rust_kzg_blst_sppark::multi_scalar_mult(&affines[0..len], &scalars)
        };

        *out = FsG1(point);
    }

    #[cfg(not(feature = "sppark"))]
    {
        use crate::types::g1::FsG1ProjAddAffine;

        *out = kzg::msm::msm_impls::msm::<FsG1, FsFp, FsG1Affine, FsG1ProjAddAffine, FsFr>(
            points,
            scalars,
            len,
            precomputation,
        );
    }
}

pub static PRECOMP: OnceLock<Vec<Vec<blst_p1_affine>>> = OnceLock::new();

pub fn g1_linear_combination_batch(_points: &[Vec<FsG1>], coeffs: &[Vec<FsFr>], _precomputation: Option<&PrecomputationTable<FsFr, FsG1, FsFp, FsG1Affine>>) -> Result<Vec<FsG1>, String> {
    
    #[cfg(feature = "parallel")]
    {
        use kzg::msm::thread_pool::ThreadPoolExt;
        
        let pool = kzg::msm::thread_pool::da_pool();
        let ncpus = pool.max_count();
        let counter = Arc::new(AtomicUsize::new(0));
        let mut results: Vec<Cell<FsG1>> = Vec::with_capacity(eth::CELLS_PER_EXT_BLOB);
        #[allow(clippy::uninit_vec)]
        unsafe {
            results.set_len(eth::CELLS_PER_EXT_BLOB);
        };
        let results = &results[..];

        for _ in 0..ncpus {
            let counter = counter.clone();
            pool.joined_execute(move || {
                let mut scalars = vec![blst_scalar::default(); eth::FIELD_ELEMENTS_PER_CELL];
                let mut scratch = vec![0u64; unsafe { (blst::blst_p1s_mult_wbits_scratch_sizeof(eth::FIELD_ELEMENTS_PER_CELL) + 7) / 8 }];

                loop {
                    let work = counter.fetch_add(1, Ordering::Relaxed);

                    if work >= eth::CELLS_PER_EXT_BLOB {
                        break;
                    }

                    for j in 0..eth::FIELD_ELEMENTS_PER_CELL {
                        unsafe { blst::blst_scalar_from_fr(&mut scalars[j], &coeffs[work][j].0); }
                    }

                    let scalars_arg = [scalars.as_ptr() as *const u8, core::ptr::null()];
                    let mut p: blst::blst_p1 = blst::blst_p1::default();
                    unsafe { blst::blst_p1s_mult_wbits(&mut p, PRECOMP.get().map(|it| it[work].as_ptr()).unwrap(), 8, eth::FIELD_ELEMENTS_PER_CELL, scalars_arg.as_ptr(), 255, scratch.as_mut_ptr()) };
                    unsafe { *results[work].as_ptr().as_mut().unwrap() = FsG1(p) };
                }
            });
        }

        pool.join();

        Ok(results.iter().map(|it| *it.as_mut()).collect())
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut results: Vec<FsG1> = Vec::with_capacity(eth::CELLS_PER_EXT_BLOB);
        let mut scalars = vec![blst_scalar::default(); eth::FIELD_ELEMENTS_PER_CELL];
        let mut scratch = vec![0u64; unsafe { (blst::blst_p1s_mult_wbits_scratch_sizeof(eth::FIELD_ELEMENTS_PER_CELL) + 7) / 8 }];

        for i in 0..eth::CELLS_PER_EXT_BLOB {

            for j in 0..eth::FIELD_ELEMENTS_PER_CELL {
                unsafe { blst::blst_scalar_from_fr(&mut scalars[j], &coeffs[i][j].0); }
            }

            let scalars_arg = [scalars.as_ptr() as *const u8, core::ptr::null()];
            let mut p: blst::blst_p1 = blst::blst_p1::default();
            unsafe { blst::blst_p1s_mult_wbits(&mut p, PRECOMP.get().map(|it| it[i].as_ptr()).unwrap(), 8, eth::FIELD_ELEMENTS_PER_CELL, scalars_arg.as_ptr(), 255, scratch.as_mut_ptr()) };
            results.push(FsG1(p));
        }

        Ok(results)
    }
}

pub fn pairings_verify(a1: &FsG1, a2: &FsG2, b1: &FsG1, b2: &FsG2) -> bool {
    let mut aa1 = blst_p1_affine::default();
    let mut bb1 = blst_p1_affine::default();

    let mut aa2 = blst_p2_affine::default();
    let mut bb2 = blst_p2_affine::default();

    // As an optimisation, we want to invert one of the pairings,
    // so we negate one of the points.
    let mut a1neg: FsG1 = *a1;
    unsafe {
        blst_p1_cneg(&mut a1neg.0, true);
        blst_p1_to_affine(&mut aa1, &a1neg.0);

        blst_p1_to_affine(&mut bb1, &b1.0);
        blst_p2_to_affine(&mut aa2, &a2.0);
        blst_p2_to_affine(&mut bb2, &b2.0);

        let dst = [0u8; 3];
        let mut pairing_blst = Pairing::new(false, &dst);
        pairing_blst.raw_aggregate(&aa2, &aa1);
        pairing_blst.raw_aggregate(&bb2, &bb1);
        let gt_point = pairing_blst.as_fp12().final_exp();

        blst_fp12_is_one(&gt_point)
    }
}
