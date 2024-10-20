use super::{Fp, P1};
use crate::kzg_proofs::{LFFTSettings, LKZGSettings};
use crate::kzg_types::{ArkFp, ArkFr, ArkG1, ArkG1Affine, ArkG2};
use crate::P2;
use ark_bls12_381::{g1, g2, Fq, Fq2, Fr as BLS12FR};
use ark_ec::models::short_weierstrass::Projective;
use ark_ff::Fp2;
use ark_poly::univariate::DensePolynomial as DensePoly;
use ark_poly::DenseUVPolynomial;
use blst::{blst_fp, blst_fp2, blst_fr, blst_p1, blst_p2};
use kzg::eip_4844::{
    Blob, CKZGSettings, PrecomputationTableManager, BYTES_PER_FIELD_ELEMENT, C_KZG_RET,
    C_KZG_RET_BADARGS, FIELD_ELEMENTS_PER_BLOB, FIELD_ELEMENTS_PER_CELL,
    FIELD_ELEMENTS_PER_EXT_BLOB, TRUSTED_SETUP_NUM_G2_POINTS,
};

use kzg::Fr;

#[derive(Debug, PartialEq, Eq)]
pub struct Error;

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct PolyData {
    pub coeffs: Vec<ArkFr>,
}
// FIXME: Store just dense poly here

//pub fn pc_poly_into_blst_poly(poly: DensePoly<Fr>) -> PolyData {
//    let mut bls_pol: Vec<ArkFr> = { Vec::new() };
//    for x in poly.coeffs {
//        bls_pol.push(ArkFr { fr: x });
//    }
//    PolyData { coeffs: bls_pol }
//}

//pub fn blst_poly_into_pc_poly(pd: &[ArkFr]) -> DensePoly<Fr> {
//    let mut poly: Vec<Fr> = vec![Fr::default(); pd.len()];
//    for i in 0..pd.len() {
//        poly[i] = pd[i].fr;
//    }
//    DensePoly::from_coefficients_vec(poly)
//}

pub const fn pc_fq_into_blst_fp(fq: Fq) -> Fp {
    Fp { l: fq.0 .0 }
}

pub const fn blst_fr_into_pc_fr(fr: blst_fr) -> BLS12FR {
    BLS12FR {
        0: ark_ff::BigInt(fr.l),
        1: core::marker::PhantomData,
    }
}

pub const fn pc_fr_into_blst_fr(fr: Fr) -> blst_fr {
    blst::blst_fr { l: fr.0 .0 }
}

pub const fn blst_fp_into_pc_fq(fp: &Fp) -> Fq {
    Fq {
        0: ark_ff::BigInt(fp.l),
        1: core::marker::PhantomData,
    }
}

pub const fn blst_fp2_into_pc_fq2(fp: &blst_fp2) -> Fq2 {
    Fp2 {
        c0: blst_fp_into_pc_fq(&fp.fp[0]),
        c1: blst_fp_into_pc_fq(&fp.fp[1]),
    }
}

pub const fn blst_p1_into_pc_g1projective(p1: &P1) -> Projective<g1::Config> {
    Projective {
        x: blst_fp_into_pc_fq(&p1.x),
        y: blst_fp_into_pc_fq(&p1.y),
        z: blst_fp_into_pc_fq(&p1.z),
    }
}

pub const fn pc_g1projective_into_blst_p1(p1: Projective<g1::Config>) -> blst_p1 {
    blst_p1 {
        x: blst_fp { l: p1.x.0 .0 },
        y: blst_fp { l: p1.y.0 .0 },
        z: blst_fp { l: p1.z.0 .0 },
    }
}

pub const fn blst_p2_into_pc_g2projective(p2: &P2) -> Projective<g2::Config> {
    Projective {
        x: blst_fp2_into_pc_fq2(&p2.x),
        y: blst_fp2_into_pc_fq2(&p2.y),
        z: blst_fp2_into_pc_fq2(&p2.z),
    }
}

pub const fn pc_g2projective_into_blst_p2(p2: Projective<g2::Config>) -> blst_p2 {
    blst_p2 {
        x: blst::blst_fp2 {
            fp: [
                blst::blst_fp { l: p2.x.c0.0 .0 },
                blst::blst_fp { l: p2.x.c1.0 .0 },
            ],
        },
        y: blst::blst_fp2 {
            fp: [
                blst::blst_fp { l: p2.y.c0.0 .0 },
                blst::blst_fp { l: p2.y.c1.0 .0 },
            ],
        },
        z: blst::blst_fp2 {
            fp: [
                blst::blst_fp { l: p2.z.c0.0 .0 },
                blst::blst_fp { l: p2.z.c1.0 .0 },
            ],
        },
    }
}

pub(crate) fn fft_settings_to_rust(
    c_settings: *const CKZGSettings,
) -> Result<LFFTSettings, String> {
    let settings = unsafe { &*c_settings };

    let roots_of_unity = unsafe {
        core::slice::from_raw_parts(settings.roots_of_unity, FIELD_ELEMENTS_PER_EXT_BLOB + 1)
            .iter()
            .map(|r| ArkFr(*r))
            .collect::<Vec<ArkFr>>()
    };

    let brp_roots_of_unity = unsafe {
        core::slice::from_raw_parts(settings.brp_roots_of_unity, FIELD_ELEMENTS_PER_EXT_BLOB)
            .iter()
            .map(|r| ArkFr(*r))
            .collect::<Vec<ArkFr>>()
    };

    let reverse_roots_of_unity = unsafe {
        core::slice::from_raw_parts(
            settings.reverse_roots_of_unity,
            FIELD_ELEMENTS_PER_EXT_BLOB + 1,
        )
        .iter()
        .map(|r| ArkFr(*r))
        .collect::<Vec<ArkFr>>()
    };

    Ok(LFFTSettings {
        max_width: FIELD_ELEMENTS_PER_EXT_BLOB,
        root_of_unity: roots_of_unity[0],
        roots_of_unity,
        brp_roots_of_unity,
        reverse_roots_of_unity,
    })
}

pub(crate) static mut PRECOMPUTATION_TABLES: PrecomputationTableManager<
    ArkFr,
    ArkG1,
    ArkFp,
    ArkG1Affine,
> = PrecomputationTableManager::new();

pub(crate) fn kzg_settings_to_rust(c_settings: &CKZGSettings) -> Result<LKZGSettings, String> {
    Ok(LKZGSettings {
        fs: fft_settings_to_rust(c_settings)?,
        secret_g1: vec![],
        g1_values_monomial: unsafe {
            core::slice::from_raw_parts(c_settings.g1_values_monomial, FIELD_ELEMENTS_PER_BLOB)
        }
        .iter()
        .map(|r| ArkG1(*r))
        .collect::<Vec<_>>(),
        g1_values_lagrange_brp: unsafe {
            core::slice::from_raw_parts(c_settings.g1_values_lagrange_brp, FIELD_ELEMENTS_PER_BLOB)
        }
        .iter()
        .map(|r| ArkG1(*r))
        .collect::<Vec<_>>(),
        secret_g2: vec![],
        g2_values_monomial: unsafe {
            core::slice::from_raw_parts(c_settings.g2_values_monomial, TRUSTED_SETUP_NUM_G2_POINTS)
        }
        .iter()
        .map(|r| ArkG2(*r))
        .collect::<Vec<_>>(),
        x_ext_fft_columns: unsafe {
            core::slice::from_raw_parts(
                c_settings.x_ext_fft_columns,
                2 * ((FIELD_ELEMENTS_PER_EXT_BLOB / 2) / FIELD_ELEMENTS_PER_CELL),
            )
        }
        .iter()
        .map(|it| {
            unsafe { core::slice::from_raw_parts(*it, FIELD_ELEMENTS_PER_CELL) }
                .iter()
                .map(|it| ArkG1(*it))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>(),
        precomputation: unsafe { PRECOMPUTATION_TABLES.get_precomputation(c_settings) },
    })
}

pub(crate) unsafe fn deserialize_blob(blob: *const Blob) -> Result<Vec<ArkFr>, C_KZG_RET> {
    (*blob)
        .bytes
        .chunks(BYTES_PER_FIELD_ELEMENT)
        .map(|chunk| {
            let mut bytes = [0u8; BYTES_PER_FIELD_ELEMENT];
            bytes.copy_from_slice(chunk);
            if let Ok(result) = ArkFr::from_bytes(&bytes) {
                Ok(result)
            } else {
                Err(C_KZG_RET_BADARGS)
            }
        })
        .collect::<Result<Vec<ArkFr>, C_KZG_RET>>()
}
