#![allow(non_camel_case_types)]

extern crate alloc;
use super::utils::{blst_poly_into_pc_poly, PolyData};
use crate::consts::{G1_GENERATOR, G2_GENERATOR};
use crate::kzg_types::{ArkFp, ArkFr, ArkG1Affine, ArkG1, ArkG2, ArkG1ProjAddAffine};
use alloc::sync::Arc;
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_poly::Polynomial;
use ark_std::{vec, One};
use kzg::eip_4844::hash_to_bls_field;
use kzg::msm::{msm_impls::msm, precompute::PrecomputationTable};
use kzg::Fr as FrTrait;
use kzg::{G1Mul, G2Mul};
use std::ops::Neg;

#[derive(Debug, Clone)]
pub struct FFTSettings {
    pub max_width: usize,
    pub root_of_unity: ArkFr,
    pub expanded_roots_of_unity: Vec<ArkFr>,
    pub reverse_roots_of_unity: Vec<ArkFr>,
    pub roots_of_unity: Vec<ArkFr>,
}

pub fn g1_linear_combination(
    out: &mut ArkG1,
    points: &[ArkG1],
    scalars: &[ArkFr],
    len: usize,
    precomputation: Option<&PrecomputationTable<ArkFr, ArkG1, ArkFp, ArkG1Affine>>,
) {
    #[cfg(feature = "sppark")]
    {
        use blst::{blst_fr, blst_scalar, blst_scalar_from_fr};
        use kzg::{G1Mul, G1};

        if len < 8 {
            *out = ArkG1::default();
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
            let affines = kzg::msm::msm_impls::batch_convert::<ArkG1, ArkFp, ArkG1Affine>(&points);
            let affines = unsafe {
                alloc::slice::from_raw_parts(affines.as_ptr() as *const blst_p1_affine, len)
            };
            rust_kzg_blst_sppark::multi_scalar_mult(&affines[0..len], &scalars)
        };

        *out = ArkG1(point);
    }

    #[cfg(not(feature = "sppark"))]
    {
        *out = msm::<ArkG1, ArkFp, ArkG1Affine, ArkG1ProjAddAffine, ArkFr>(
            points,
            scalars,
            len,
            precomputation,
        );
    }
}

pub fn expand_root_of_unity(root: &ArkFr, width: usize) -> Result<Vec<ArkFr>, String> {
    let mut generated_powers = vec![ArkFr::one(), *root];

    while !(generated_powers.last().unwrap().is_one()) {
        if generated_powers.len() > width {
            return Err(String::from("Root of unity multiplied for too long"));
        }

        generated_powers.push(generated_powers.last().unwrap().mul(root));
    }

    if generated_powers.len() != width + 1 {
        return Err(String::from("Root of unity has invalid scale"));
    }

    Ok(generated_powers)
}

#[derive(Debug, Clone, Default)]
pub struct KZGSettings {
    pub fs: FFTSettings,
    pub secret_g1: Vec<ArkG1>,
    pub secret_g2: Vec<ArkG2>,
    pub precomputation: Option<Arc<PrecomputationTable<ArkFr, ArkG1, ArkFp, ArkG1Affine>>>,
}

pub fn generate_trusted_setup(len: usize, secret: [u8; 32usize]) -> (Vec<ArkG1>, Vec<ArkG2>) {
    let s = hash_to_bls_field::<ArkFr>(&secret);
    let mut s_pow = ArkFr::one();

    let mut s1 = Vec::with_capacity(len);
    let mut s2 = Vec::with_capacity(len);

    for _ in 0..len {
        s1.push(G1_GENERATOR.mul(&s_pow));
        s2.push(G2_GENERATOR.mul(&s_pow));

        s_pow = s_pow.mul(&s);
    }

    (s1, s2)
}

pub fn eval_poly(p: &PolyData, x: &ArkFr) -> ArkFr {
    let poly = blst_poly_into_pc_poly(&p.coeffs);
    ArkFr {
        fr: poly.evaluate(&x.fr),
    }
}

pub fn pairings_verify(a1: &ArkG1, a2: &ArkG2, b1: &ArkG1, b2: &ArkG2) -> bool {
    let ark_a1_neg = a1.0.neg().into_affine();
    let ark_b1 = b1.0.into_affine();
    let ark_a2 = a2.0.into_affine();
    let ark_b2 = b2.0.into_affine();

    Bls12_381::multi_pairing([ark_a1_neg, ark_b1], [ark_a2, ark_b2])
        .0
        .is_one()
}
