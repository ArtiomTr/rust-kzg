use crate::consts::{
    G1_GENERATOR, G1_IDENTITY, G1_NEGATIVE_GENERATOR, G2_GENERATOR, G2_NEGATIVE_GENERATOR,
    SCALE2_ROOT_OF_UNITY, SCALE_FACTOR,
};
use crate::fft_g1::{fft_g1_fast, g1_linear_combination};
use crate::kzg_proofs::{expand_root_of_unity, pairings_verify, LFFTSettings, LKZGSettings};
use crate::recover::{scale_poly, unscale_poly};
use crate::utils::{
    blst_fp_into_pc_fq, blst_fr_into_pc_fr, blst_p1_into_pc_g1projective,
    blst_p2_into_pc_g2projective, pc_fr_into_blst_fr, pc_g1projective_into_blst_p1,
    pc_g2projective_into_blst_p2, PolyData,
};
use ark_bls12_381::{g1, g2, Fr, G1Affine, G2Affine};
use ark_ec::{models::short_weierstrass::Projective, AffineRepr, Group};
use ark_ec::{CurveConfig, CurveGroup};
use ark_ff::{biginteger::BigInteger256, BigInteger, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};

#[cfg(feature = "rand")]
use ark_std::UniformRand;

use blst::p1_affines;
use blst::{
    blst_bendian_from_scalar, blst_fp, blst_fp2, blst_fr, blst_fr_add, blst_fr_cneg,
    blst_fr_eucl_inverse, blst_fr_from_scalar, blst_fr_from_uint64, blst_fr_inverse, blst_fr_mul,
    blst_fr_sqr, blst_fr_sub, blst_p1, blst_p1_add, blst_p1_add_or_double, blst_p1_affine,
    blst_p1_cneg, blst_p1_compress, blst_p1_double, blst_p1_from_affine, blst_p1_in_g1,
    blst_p1_is_equal, blst_p1_is_inf, blst_p1_mult, blst_p1_uncompress, blst_p2,
    blst_p2_add_or_double, blst_p2_affine, blst_p2_cneg, blst_p2_compress, blst_p2_double,
    blst_p2_from_affine, blst_p2_is_equal, blst_p2_mult, blst_p2_uncompress, blst_scalar,
    blst_scalar_fr_check, blst_scalar_from_bendian, blst_scalar_from_fr, blst_uint64_from_fr,
    BLST_ERROR,
};
use kzg::common_utils::{log2_u64, log_2_byte, reverse_bit_order};
use kzg::eip_4844::{
    BYTES_PER_FIELD_ELEMENT, BYTES_PER_G1, BYTES_PER_G2, FIELD_ELEMENTS_PER_BLOB,
    FIELD_ELEMENTS_PER_CELL, FIELD_ELEMENTS_PER_EXT_BLOB, TRUSTED_SETUP_NUM_G2_POINTS,
};
use kzg::msm::precompute::{precompute, PrecomputationTable};
use kzg::{
    FFTFr, FFTSettings, FFTSettingsPoly, Fr as KzgFr, G1Affine as G1AffineTrait, G1Fp, G1GetFp,
    G1LinComb, G1Mul, G1ProjAddAffine, G2Mul, KZGSettings, PairingVerify, Poly, Scalar256, G1, G2,
};
use std::ops::{AddAssign, Mul, Neg, Sub};

extern crate alloc;
use alloc::sync::Arc;
use std::ptr;

//#[derive(Debug, Clone, Default)]
//pub struct ArkKZGSettings {
//    pub fs: ArkFFTSettings,
//    pub g1_values_monomial: Vec<ArkG1>,
//    pub g1_values_lagrange_brp: Vec<ArkG1>,
//    pub g2_values_monomial: Vec<ArkG2>,
//    pub precomputation: Option<Arc<PrecomputationTable<ArkFr, ArkG1, ArkFp, ArkG1Affine>>>,
//}

fn bytes_be_to_uint64(inp: &[u8]) -> u64 {
    u64::from_be_bytes(inp.try_into().expect("Input wasn't 8 elements..."))
}

const BLS12_381_MOD_256: [u64; 4] = [
    0xffffffff00000001,
    0x53bda402fffe5bfe,
    0x3339d80809a1d805,
    0x73eda753299d7d48,
];

#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub struct ArkFr(pub blst_fr);

impl Default for LFFTSettings {
    fn default() -> Self {
        Self::new(0).unwrap()
    }
}

fn bigint_check_mod_256(a: &[u64; 4]) -> bool {
    let (_, overflow) = a[0].overflowing_sub(BLS12_381_MOD_256[0]);
    let (_, overflow) = a[1].overflowing_sub(BLS12_381_MOD_256[1] + overflow as u64);
    let (_, overflow) = a[2].overflowing_sub(BLS12_381_MOD_256[2] + overflow as u64);
    let (_, overflow) = a[3].overflowing_sub(BLS12_381_MOD_256[3] + overflow as u64);
    overflow
}

impl KzgFr for ArkFr {
    fn null() -> Self {
        Self::from_u64_arr(&[u64::MAX, u64::MAX, u64::MAX, u64::MAX])
    }

    fn zero() -> Self {
        Self::from_u64(0)
    }

    fn one() -> Self {
        Self::from_u64(1)
    }

    #[cfg(feature = "rand")]
    fn rand() -> Self {
        let val: [u64; 4] = [
            rand::random(),
            rand::random(),
            rand::random(),
            rand::random(),
        ];
        let mut ret = Self::default();
        unsafe {
            blst_fr_from_uint64(&mut ret.0, val.as_ptr());
        }

        ret
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_FIELD_ELEMENT,
                    bytes.len()
                )
            })
            .and_then(|bytes: &[u8; BYTES_PER_FIELD_ELEMENT]| {
                let mut bls_scalar = blst_scalar::default();
                let mut fr = blst_fr::default();
                unsafe {
                    blst_scalar_from_bendian(&mut bls_scalar, bytes.as_ptr());
                    if !blst_scalar_fr_check(&bls_scalar) {
                        return Err("Invalid scalar".to_string());
                    }
                    blst_fr_from_scalar(&mut fr, &bls_scalar);
                }
                Ok(Self(fr))
            })
    }

    fn from_bytes_unchecked(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_FIELD_ELEMENT,
                    bytes.len()
                )
            })
            .map(|bytes: &[u8; BYTES_PER_FIELD_ELEMENT]| {
                let mut bls_scalar = blst_scalar::default();
                let mut fr = blst_fr::default();
                unsafe {
                    blst_scalar_from_bendian(&mut bls_scalar, bytes.as_ptr());
                    blst_fr_from_scalar(&mut fr, &bls_scalar);
                }
                Self(fr)
            })
    }

    fn from_hex(hex: &str) -> Result<Self, String> {
        let bytes = hex::decode(&hex[2..]).unwrap();
        Self::from_bytes(&bytes)
    }

    fn from_u64_arr(u: &[u64; 4]) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_from_uint64(&mut ret.0, u.as_ptr());
        }

        ret
    }

    fn from_u64(val: u64) -> Self {
        Self::from_u64_arr(&[val, 0, 0, 0])
    }

    fn to_bytes(&self) -> [u8; 32] {
        let mut scalar = blst_scalar::default();
        let mut bytes = [0u8; 32];
        unsafe {
            blst_scalar_from_fr(&mut scalar, &self.0);
            blst_bendian_from_scalar(bytes.as_mut_ptr(), &scalar);
        }

        bytes
    }

    fn to_u64_arr(&self) -> [u64; 4] {
        let mut val: [u64; 4] = [0; 4];
        unsafe {
            blst_uint64_from_fr(val.as_mut_ptr(), &self.0);
        }

        val
    }

    fn is_one(&self) -> bool {
        let mut val: [u64; 4] = [0; 4];
        unsafe {
            blst_uint64_from_fr(val.as_mut_ptr(), &self.0);
        }

        val[0] == 1 && val[1] == 0 && val[2] == 0 && val[3] == 0
    }

    fn is_zero(&self) -> bool {
        let mut val: [u64; 4] = [0; 4];
        unsafe {
            blst_uint64_from_fr(val.as_mut_ptr(), &self.0);
        }

        val[0] == 0 && val[1] == 0 && val[2] == 0 && val[3] == 0
    }

    fn is_null(&self) -> bool {
        self.equals(&Self::null())
    }

    fn sqr(&self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_sqr(&mut ret.0, &self.0);
        }

        ret
    }

    fn mul(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_mul(&mut ret.0, &self.0, &b.0);
        }

        ret
    }

    fn add(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_add(&mut ret.0, &self.0, &b.0);
        }

        ret
    }

    fn sub(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_sub(&mut ret.0, &self.0, &b.0);
        }

        ret
    }

    fn eucl_inverse(&self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_eucl_inverse(&mut ret.0, &self.0);
        }

        ret
    }

    fn negate(&self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_cneg(&mut ret.0, &self.0, true);
        }

        ret
    }

    fn inverse(&self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_fr_inverse(&mut ret.0, &self.0);
        }

        ret
    }

    fn pow(&self, n: usize) -> Self {
        let mut out = Self::one();

        let mut temp = *self;
        let mut n = n;
        loop {
            if (n & 1) == 1 {
                out = out.mul(&temp);
            }
            n >>= 1;
            if n == 0 {
                break;
            }

            temp = temp.sqr();
        }

        out
    }

    fn div(&self, b: &Self) -> Result<Self, String> {
        let tmp = b.eucl_inverse();
        let out = self.mul(&tmp);

        Ok(out)
    }

    fn equals(&self, b: &Self) -> bool {
        let mut val_a: [u64; 4] = [0; 4];
        let mut val_b: [u64; 4] = [0; 4];

        unsafe {
            blst_uint64_from_fr(val_a.as_mut_ptr(), &self.0);
            blst_uint64_from_fr(val_b.as_mut_ptr(), &b.0);
        }

        val_a[0] == val_b[0] && val_a[1] == val_b[1] && val_a[2] == val_b[2] && val_a[3] == val_b[3]
    }

    fn to_scalar(&self) -> Scalar256 {
        let mut blst_scalar = blst_scalar::default();
        unsafe {
            blst_scalar_from_fr(&mut blst_scalar, &self.0);
        }
        Scalar256::from_u8(&blst_scalar.b)
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ArkG1(pub blst_p1);

impl ArkG1 {
    pub(crate) const fn from_xyz(x: blst_fp, y: blst_fp, z: blst_fp) -> Self {
        ArkG1(blst_p1 { x, y, z })
    }
}

impl G1 for ArkG1 {
    fn identity() -> Self {
        G1_IDENTITY
    }

    fn generator() -> Self {
        G1_GENERATOR
    }

    fn negative_generator() -> Self {
        G1_NEGATIVE_GENERATOR
    }

    #[cfg(feature = "rand")]
    fn rand() -> Self {
        let result: ArkG1 = G1_GENERATOR;
        result.mul(&kzg::Fr::rand())
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_G1,
                    bytes.len()
                )
            })
            .and_then(|bytes: &[u8; BYTES_PER_G1]| {
                let mut tmp = blst_p1_affine::default();
                let mut g1 = blst_p1::default();
                unsafe {
                    // The uncompress routine also checks that the point is on the curve
                    if blst_p1_uncompress(&mut tmp, bytes.as_ptr()) != BLST_ERROR::BLST_SUCCESS {
                        return Err("Failed to uncompress".to_string());
                    }
                    blst_p1_from_affine(&mut g1, &tmp);
                }
                Ok(ArkG1(g1))
            })
    }

    fn from_hex(hex: &str) -> Result<Self, String> {
        let bytes = hex::decode(&hex[2..]).unwrap();
        Self::from_bytes(&bytes)
    }

    fn to_bytes(&self) -> [u8; 48] {
        let mut out = [0u8; BYTES_PER_G1];
        unsafe {
            blst_p1_compress(out.as_mut_ptr(), &self.0);
        }
        out
    }

    fn add_or_dbl(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_p1_add_or_double(&mut ret.0, &self.0, &b.0);
        }
        ret
    }

    fn is_inf(&self) -> bool {
        unsafe { blst_p1_is_inf(&self.0) }
    }

    fn is_valid(&self) -> bool {
        unsafe {
            // The point must be on the right subgroup
            blst_p1_in_g1(&self.0)
        }
    }

    fn dbl(&self) -> Self {
        let mut result = blst_p1::default();
        unsafe {
            blst_p1_double(&mut result, &self.0);
        }
        Self(result)
    }

    fn add(&self, b: &Self) -> Self {
        let mut ret = Self::default();
        unsafe {
            blst_p1_add(&mut ret.0, &self.0, &b.0);
        }
        ret
    }

    fn sub(&self, b: &Self) -> Self {
        let mut b_negative: ArkG1 = *b;
        let mut ret = Self::default();
        unsafe {
            blst_p1_cneg(&mut b_negative.0, true);
            blst_p1_add_or_double(&mut ret.0, &self.0, &b_negative.0);
            ret
        }
    }

    fn equals(&self, b: &Self) -> bool {
        unsafe { blst_p1_is_equal(&self.0, &b.0) }
    }

    fn zero() -> Self {
        Self(blst_p1 {
            x: blst_fp {
                l: [
                    8505329371266088957,
                    17002214543764226050,
                    6865905132761471162,
                    8632934651105793861,
                    6631298214892334189,
                    1582556514881692819,
                ],
            },
            y: blst_fp {
                l: [
                    8505329371266088957,
                    17002214543764226050,
                    6865905132761471162,
                    8632934651105793861,
                    6631298214892334189,
                    1582556514881692819,
                ],
            },
            z: blst_fp {
                l: [0, 0, 0, 0, 0, 0],
            },
        })
    }

    fn add_or_dbl_assign(&mut self, b: &Self) {
        unsafe {
            blst::blst_p1_add_or_double(&mut self.0, &self.0, &b.0);
        }
    }

    fn add_assign(&mut self, b: &Self) {
        unsafe {
            blst::blst_p1_add(&mut self.0, &self.0, &b.0);
        }
    }

    fn dbl_assign(&mut self) {
        unsafe {
            blst::blst_p1_double(&mut self.0, &self.0);
        }
    }
}

impl G1GetFp<ArkFp> for ArkG1 {
    fn x(&self) -> &ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&self.0.x)
        }
    }

    fn y(&self) -> &ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&self.0.y)
        }
    }

    fn z(&self) -> &ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&self.0.z)
        }
    }

    fn x_mut(&mut self) -> &mut ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&mut self.0.x)
        }
    }

    fn y_mut(&mut self) -> &mut ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&mut self.0.y)
        }
    }

    fn z_mut(&mut self) -> &mut ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&mut self.0.z)
        }
    }
}

impl G1Mul<ArkFr> for ArkG1 {
    fn mul(&self, b: &ArkFr) -> Self {
        let mut scalar: blst_scalar = blst_scalar::default();
        unsafe {
            blst_scalar_from_fr(&mut scalar, &b.0);
        }

        // Count the number of bytes to be multiplied.
        let mut i = scalar.b.len();
        while i != 0 && scalar.b[i - 1] == 0 {
            i -= 1;
        }

        let mut result = Self::default();
        if i == 0 {
            return G1_IDENTITY;
        } else if i == 1 && scalar.b[0] == 1 {
            return *self;
        } else {
            // Count the number of bits to be multiplied.
            unsafe {
                blst_p1_mult(
                    &mut result.0,
                    &self.0,
                    &(scalar.b[0]),
                    8 * i - 7 + log_2_byte(scalar.b[i - 1]),
                );
            }
        }
        result
    }
}

impl G1LinComb<ArkFr, ArkFp, ArkG1Affine> for ArkG1 {
    fn g1_lincomb(
        points: &[Self],
        scalars: &[ArkFr],
        len: usize,
        precomputation: Option<&PrecomputationTable<ArkFr, Self, ArkFp, ArkG1Affine>>,
    ) -> Self {
        let mut out = ArkG1::default();
        g1_linear_combination(&mut out, points, scalars, len, precomputation);
        out
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ArkG1Affine(pub blst_p1_affine);

impl kzg::G1Affine<ArkG1, ArkFp> for ArkG1Affine {
    fn zero() -> Self {
        Self(blst_p1_affine {
            x: {
                blst_fp {
                    l: [0, 0, 0, 0, 0, 0],
                }
            },
            y: {
                blst_fp {
                    l: [0, 0, 0, 0, 0, 0],
                }
            },
        })
    }

    fn into_affine(g1: &ArkG1) -> Self {
        let mut ret: Self = Default::default();
        unsafe {
            blst::blst_p1_to_affine(&mut ret.0, &g1.0);
        }
        ret
    }

    fn into_affines_loc(out: &mut [Self], g1: &[ArkG1]) {
        let p: [*const blst_p1; 2] = [g1.as_ptr() as *const blst_p1, ptr::null()];
        unsafe {
            blst::blst_p1s_to_affine(out.as_mut_ptr() as *mut blst_p1_affine, &p[0], g1.len());
        }
    }

    fn into_affines(g1: &[ArkG1]) -> Vec<Self> {
        let points =
            unsafe { core::slice::from_raw_parts(g1.as_ptr() as *const blst_p1, g1.len()) };
        let points = p1_affines::from(points);
        unsafe {
            // Transmute safe due to repr(C) on ArkG1Affine
            core::mem::transmute(points)
        }
    }

    fn to_proj(&self) -> ArkG1 {
        let mut ret: ArkG1 = Default::default();
        unsafe {
            blst::blst_p1_from_affine(&mut ret.0, &self.0);
        }
        ret
    }

    fn x(&self) -> &ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&self.0.x)
        }
    }

    fn y(&self) -> &ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&self.0.y)
        }
    }

    fn is_infinity(&self) -> bool {
        unsafe { blst::blst_p1_affine_is_inf(&self.0) }
    }

    fn x_mut(&mut self) -> &mut ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&mut self.0.x)
        }
    }

    fn y_mut(&mut self) -> &mut ArkFp {
        unsafe {
            // Transmute safe due to repr(C) on ArkFp
            core::mem::transmute(&mut self.0.y)
        }
    }
}

impl PairingVerify<ArkG1, ArkG2> for ArkG1 {
    fn verify(a1: &ArkG1, a2: &ArkG2, b1: &ArkG1, b2: &ArkG2) -> bool {
        pairings_verify(a1, a2, b1, b2)
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ArkG2(pub blst_p2);

impl ArkG2 {
    pub(crate) fn _from_xyz(x: blst_fp2, y: blst_fp2, z: blst_fp2) -> Self {
        ArkG2(blst_p2 { x, y, z })
    }

    #[cfg(feature = "rand")]
    pub fn rand() -> Self {
        let result: ArkG2 = G2_GENERATOR;
        result.mul(&ArkFr::rand())
    }
}

impl G2 for ArkG2 {
    fn generator() -> Self {
        G2_GENERATOR
    }

    fn negative_generator() -> Self {
        G2_NEGATIVE_GENERATOR
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_G2,
                    bytes.len()
                )
            })
            .and_then(|bytes: &[u8; BYTES_PER_G2]| {
                let mut tmp = blst_p2_affine::default();
                let mut g2 = blst_p2::default();
                unsafe {
                    // The uncompress routine also checks that the point is on the curve
                    if blst_p2_uncompress(&mut tmp, bytes.as_ptr()) != BLST_ERROR::BLST_SUCCESS {
                        return Err("Failed to uncompress".to_string());
                    }
                    blst_p2_from_affine(&mut g2, &tmp);
                }
                Ok(ArkG2(g2))
            })
    }

    fn to_bytes(&self) -> [u8; 96] {
        let mut out = [0u8; BYTES_PER_G2];
        unsafe {
            blst_p2_compress(out.as_mut_ptr(), &self.0);
        }
        out
    }

    fn add_or_dbl(&mut self, b: &Self) -> Self {
        let mut result = blst_p2::default();
        unsafe {
            blst_p2_add_or_double(&mut result, &self.0, &b.0);
        }
        Self(result)
    }

    fn dbl(&self) -> Self {
        let mut result = blst_p2::default();
        unsafe {
            blst_p2_double(&mut result, &self.0);
        }
        Self(result)
    }

    fn sub(&self, b: &Self) -> Self {
        let mut bneg: blst_p2 = b.0;
        let mut result = blst_p2::default();
        unsafe {
            blst_p2_cneg(&mut bneg, true);
            blst_p2_add_or_double(&mut result, &self.0, &bneg);
        }
        Self(result)
    }

    fn equals(&self, b: &Self) -> bool {
        unsafe { blst_p2_is_equal(&self.0, &b.0) }
    }
}

impl G2Mul<ArkFr> for ArkG2 {
    fn mul(&self, b: &ArkFr) -> Self {
        let mut result = blst_p2::default();
        let mut scalar = blst_scalar::default();
        unsafe {
            blst_scalar_from_fr(&mut scalar, &b.0);
            blst_p2_mult(
                &mut result,
                &self.0,
                scalar.b.as_ptr(),
                8 * core::mem::size_of::<blst_scalar>(),
            );
        }
        Self(result)
    }
}

impl Poly<ArkFr> for PolyData {
    fn new(size: usize) -> Self {
        Self {
            coeffs: vec![ArkFr::default(); size],
        }
    }

    fn get_coeff_at(&self, i: usize) -> ArkFr {
        self.coeffs[i]
    }

    fn set_coeff_at(&mut self, i: usize, x: &ArkFr) {
        self.coeffs[i] = *x
    }

    fn get_coeffs(&self) -> &[ArkFr] {
        &self.coeffs
    }

    fn len(&self) -> usize {
        self.coeffs.len()
    }

    fn eval(&self, x: &ArkFr) -> ArkFr {
        if self.coeffs.is_empty() {
            return ArkFr::zero();
        } else if x.is_zero() {
            return self.coeffs[0];
        }

        let mut ret = self.coeffs[self.coeffs.len() - 1];
        let mut i = self.coeffs.len() - 2;
        loop {
            let temp = ret.mul(x);
            ret = temp.add(&self.coeffs[i]);

            if i == 0 {
                break;
            }
            i -= 1;
        }

        ret
    }

    fn scale(&mut self) {
        let scale_factor = ArkFr::from_u64(SCALE_FACTOR);
        let inv_factor = scale_factor.inverse();

        let mut factor_power = ArkFr::one();
        for i in 0..self.coeffs.len() {
            factor_power = factor_power.mul(&inv_factor);
            self.coeffs[i] = self.coeffs[i].mul(&factor_power);
        }
    }

    fn unscale(&mut self) {
        let scale_factor = ArkFr::from_u64(SCALE_FACTOR);

        let mut factor_power = ArkFr::one();
        for i in 0..self.coeffs.len() {
            factor_power = factor_power.mul(&scale_factor);
            self.coeffs[i] = self.coeffs[i].mul(&factor_power);
        }
    }

    // TODO: analyze how algo works
    fn inverse(&mut self, output_len: usize) -> Result<Self, String> {
        if output_len == 0 {
            return Err(String::from("Can't produce a zero-length result"));
        } else if self.coeffs.is_empty() {
            return Err(String::from("Can't inverse a zero-length poly"));
        } else if self.coeffs[0].is_zero() {
            return Err(String::from(
                "First coefficient of polynomial mustn't be zero",
            ));
        }

        let mut ret = PolyData {
            coeffs: vec![ArkFr::zero(); output_len],
        };
        // If the input polynomial is constant, the remainder of the series is zero
        if self.coeffs.len() == 1 {
            ret.coeffs[0] = self.coeffs[0].eucl_inverse();

            return Ok(ret);
        }

        let maxd = output_len - 1;

        // Max space for multiplications is (2 * length - 1)
        // Don't need the following as its recalculated inside
        // let scale: usize = log2_pow2(next_pow_of_2(2 * output_len - 1));
        // let fft_settings = FsFFTSettings::new(scale).unwrap();

        // To store intermediate results

        // Base case for d == 0
        ret.coeffs[0] = self.coeffs[0].eucl_inverse();
        let mut d: usize = 0;
        let mut mask: usize = 1 << log2_u64(maxd);
        while mask != 0 {
            d = 2 * d + usize::from((maxd & mask) != 0);
            mask >>= 1;

            // b.c -> tmp0 (we're using out for c)
            // tmp0.length = min_u64(d + 1, b->length + output->length - 1);
            let len_temp = (d + 1).min(self.len() + output_len - 1);
            let mut tmp0 = self.mul(&ret, len_temp).unwrap();

            // 2 - b.c -> tmp0
            for i in 0..tmp0.len() {
                tmp0.coeffs[i] = tmp0.coeffs[i].negate();
            }
            let fr_two = kzg::Fr::from_u64(2);
            tmp0.coeffs[0] = tmp0.coeffs[0].add(&fr_two);

            // c.(2 - b.c) -> tmp1;
            let tmp1 = ret.mul(&tmp0, d + 1).unwrap();

            for i in 0..tmp1.len() {
                ret.coeffs[i] = tmp1.coeffs[i];
            }
        }

        if d + 1 != output_len {
            return Err(String::from("D + 1 must be equal to output_len"));
        }

        Ok(ret)
    }

    fn div(&mut self, divisor: &Self) -> Result<Self, String> {
        if divisor.len() >= self.len() || divisor.len() < 128 {
            // Tunable parameter
            self.long_div(divisor)
        } else {
            self.fast_div(divisor)
        }
    }

    fn long_div(&mut self, divisor: &Self) -> Result<Self, String> {
        if divisor.coeffs.is_empty() {
            return Err(String::from("Can't divide by zero"));
        } else if divisor.coeffs[divisor.coeffs.len() - 1].is_zero() {
            return Err(String::from("Highest coefficient must be non-zero"));
        }

        let out_length = self.poly_quotient_length(divisor);
        if out_length == 0 {
            return Ok(PolyData { coeffs: vec![] });
        }

        // Special case for divisor.len() == 2
        if divisor.len() == 2 {
            let divisor_0 = divisor.coeffs[0];
            let divisor_1 = divisor.coeffs[1];

            let mut out_coeffs = Vec::from(&self.coeffs[1..]);
            for i in (1..out_length).rev() {
                out_coeffs[i] = out_coeffs[i].div(&divisor_1).unwrap();

                let tmp = out_coeffs[i].mul(&divisor_0);
                out_coeffs[i - 1] = out_coeffs[i - 1].sub(&tmp);
            }

            out_coeffs[0] = out_coeffs[0].div(&divisor_1).unwrap();

            Ok(PolyData { coeffs: out_coeffs })
        } else {
            let mut out: PolyData = PolyData {
                coeffs: vec![ArkFr::default(); out_length],
            };

            let mut a_pos = self.len() - 1;
            let b_pos = divisor.len() - 1;
            let mut diff = a_pos - b_pos;

            let mut a = self.coeffs.clone();

            while diff > 0 {
                out.coeffs[diff] = a[a_pos].div(&divisor.coeffs[b_pos]).unwrap();

                for i in 0..(b_pos + 1) {
                    let tmp = out.coeffs[diff].mul(&divisor.coeffs[i]);
                    a[diff + i] = a[diff + i].sub(&tmp);
                }

                diff -= 1;
                a_pos -= 1;
            }

            out.coeffs[0] = a[a_pos].div(&divisor.coeffs[b_pos]).unwrap();
            Ok(out)
        }
    }

    fn fast_div(&mut self, divisor: &Self) -> Result<Self, String> {
        if divisor.coeffs.is_empty() {
            return Err(String::from("Cant divide by zero"));
        } else if divisor.coeffs[divisor.coeffs.len() - 1].is_zero() {
            return Err(String::from("Highest coefficient must be non-zero"));
        }

        let m: usize = self.len() - 1;
        let n: usize = divisor.len() - 1;

        // If the divisor is larger than the dividend, the result is zero-length
        if n > m {
            return Ok(PolyData { coeffs: Vec::new() });
        }

        // Special case for divisor.length == 1 (it's a constant)
        if divisor.len() == 1 {
            let mut out = PolyData {
                coeffs: vec![ArkFr::zero(); self.len()],
            };
            for i in 0..out.len() {
                out.coeffs[i] = self.coeffs[i].div(&divisor.coeffs[0]).unwrap();
            }
            return Ok(out);
        }

        let mut a_flip = self.flip().unwrap();
        let mut b_flip = divisor.flip().unwrap();

        let inv_b_flip = b_flip.inverse(m - n + 1).unwrap();
        let q_flip = a_flip.mul(&inv_b_flip, m - n + 1).unwrap();

        let out = q_flip.flip().unwrap();
        Ok(out)
    }

    fn mul_direct(&mut self, multiplier: &Self, output_len: usize) -> Result<Self, String> {
        if self.len() == 0 || multiplier.len() == 0 {
            return Ok(PolyData::new(0));
        }

        let a_degree = self.len() - 1;
        let b_degree = multiplier.len() - 1;

        let mut ret = PolyData {
            coeffs: vec![kzg::Fr::zero(); output_len],
        };

        // Truncate the output to the length of the output polynomial
        for i in 0..(a_degree + 1) {
            let mut j = 0;
            while (j <= b_degree) && ((i + j) < output_len) {
                let tmp = self.coeffs[i].mul(&multiplier.coeffs[j]);
                let tmp = ret.coeffs[i + j].add(&tmp);
                ret.coeffs[i + j] = tmp;

                j += 1;
            }
        }

        Ok(ret)
    }
}

impl FFTSettingsPoly<ArkFr, PolyData, LFFTSettings> for LFFTSettings {
    fn poly_mul_fft(
        a: &PolyData,
        b: &PolyData,
        len: usize,
        _fs: Option<&LFFTSettings>,
    ) -> Result<PolyData, String> {
        b.mul_fft(a, len)
    }
}

impl FFTSettings<ArkFr> for LFFTSettings {
    /// Create FFTSettings with roots of unity for a selected scale. Resulting roots will have a magnitude of 2 ^ max_scale.
    fn new(scale: usize) -> Result<LFFTSettings, String> {
        if scale >= SCALE2_ROOT_OF_UNITY.len() {
            return Err(String::from(
                "Scale is expected to be within root of unity matrix row size",
            ));
        }

        // max_width = 2 ^ max_scale
        let max_width: usize = 1 << scale;
        let root_of_unity = ArkFr::from_u64_arr(&SCALE2_ROOT_OF_UNITY[scale]);

        // create max_width of roots & store them reversed as well
        let roots_of_unity = expand_root_of_unity(&root_of_unity, max_width)?;

        let mut brp_roots_of_unity = roots_of_unity.clone();
        brp_roots_of_unity.pop();
        reverse_bit_order(&mut brp_roots_of_unity)?;

        let mut reverse_roots_of_unity = roots_of_unity.clone();
        reverse_roots_of_unity.reverse();

        Ok(LFFTSettings {
            max_width,
            root_of_unity,
            reverse_roots_of_unity,
            roots_of_unity,
            brp_roots_of_unity,
        })
    }

    fn get_max_width(&self) -> usize {
        self.max_width
    }

    fn get_reverse_roots_of_unity_at(&self, i: usize) -> ArkFr {
        self.reverse_roots_of_unity[i]
    }

    fn get_reversed_roots_of_unity(&self) -> &[ArkFr] {
        &self.reverse_roots_of_unity
    }

    fn get_roots_of_unity_at(&self, i: usize) -> ArkFr {
        self.roots_of_unity[i]
    }

    fn get_roots_of_unity(&self) -> &[ArkFr] {
        &self.roots_of_unity
    }

    fn get_brp_roots_of_unity(&self) -> &[ArkFr] {
        &self.brp_roots_of_unity
    }

    fn get_brp_roots_of_unity_at(&self, i: usize) -> ArkFr {
        self.brp_roots_of_unity[i]
    }
}

fn g1_fft(output: &mut [ArkG1], input: &[ArkG1], s: &LFFTSettings) -> Result<(), String> {
    // g1_t *out, const g1_t *in, size_t n, const KZGSettings *s

    /* Ensure the length is valid */
    if input.len() > FIELD_ELEMENTS_PER_EXT_BLOB || !input.len().is_power_of_two() {
        return Err("Invalid input size".to_string());
    }

    let roots_stride = FIELD_ELEMENTS_PER_EXT_BLOB / input.len();
    fft_g1_fast(output, input, 1, &s.roots_of_unity, roots_stride);

    Ok(())
}

fn toeplitz_part_1(output: &mut [ArkG1], x: &[ArkG1], s: &LFFTSettings) -> Result<(), String> {
    let n = x.len();
    let n2 = n * 2;
    let mut x_ext = vec![ArkG1::identity(); n2];

    x_ext[..n].copy_from_slice(x);

    g1_fft(output, &x_ext, s)?;

    Ok(())
}

impl KZGSettings<ArkFr, ArkG1, ArkG2, LFFTSettings, PolyData, ArkFp, ArkG1Affine> for LKZGSettings {
    fn new(
        g1_monomial: &[ArkG1],
        g1_lagrange_brp: &[ArkG1],
        g2_monomial: &[ArkG2],
        fft_settings: &LFFTSettings,
    ) -> Result<Self, String> {
        if g1_monomial.len() != FIELD_ELEMENTS_PER_BLOB
            || g1_lagrange_brp.len() != FIELD_ELEMENTS_PER_BLOB
            || g2_monomial.len() != TRUSTED_SETUP_NUM_G2_POINTS
        {
            return Err("Length does not match FIELD_ELEMENTS_PER_BLOB".to_string());
        }

        let n = FIELD_ELEMENTS_PER_EXT_BLOB / 2;
        let k = n / FIELD_ELEMENTS_PER_CELL;
        let k2 = 2 * k;

        let mut points = vec![ArkG1::default(); k2];
        let mut x = vec![ArkG1::default(); k];
        let mut x_ext_fft_columns = vec![vec![ArkG1::default(); FIELD_ELEMENTS_PER_CELL]; k2];

        for offset in 0..FIELD_ELEMENTS_PER_CELL {
            let start = n - FIELD_ELEMENTS_PER_CELL - 1 - offset;
            for (i, p) in x.iter_mut().enumerate().take(k - 1) {
                let j = start - i * FIELD_ELEMENTS_PER_CELL;
                *p = g1_monomial[j];
            }
            x[k - 1] = ArkG1::identity();

            toeplitz_part_1(&mut points, &x, fft_settings)?;

            for row in 0..k2 {
                x_ext_fft_columns[row][offset] = points[row];
            }
        }

        // for (size_t offset = 0; offset < FIELD_ELEMENTS_PER_CELL; offset++) {
        //     /* Compute x, sections of the g1 values */
        //     size_t start = n - FIELD_ELEMENTS_PER_CELL - 1 - offset;
        //     for (size_t i = 0; i < k - 1; i++) {
        //         size_t j = start - i * FIELD_ELEMENTS_PER_CELL;
        //         x[i] = s->g1_values_monomial[j];
        //     }
        //     x[k - 1] = G1_IDENTITY;

        //     /* Compute points, the fft of an extended x */
        //     ret = toeplitz_part_1(points, x, k, s);
        //     if (ret != C_KZG_OK) goto out;

        //     /* Reorganize from rows into columns */
        //     for (size_t row = 0; row < k2; row++) {
        //         s->x_ext_fft_columns[row][offset] = points[row];
        //     }
        // }

        Ok(Self {
            g1_values_monomial: g1_monomial.to_vec(),
            g1_values_lagrange_brp: g1_lagrange_brp.to_vec(),
            secret_g1: vec![],
            secret_g2: vec![],
            g2_values_monomial: g2_monomial.to_vec(),
            fs: fft_settings.clone(),
            x_ext_fft_columns,
            precomputation: {
                #[cfg(feature = "sppark")]
                {
                    use blst::blst_p1_affine;
                    let points =
                        kzg::msm::msm_impls::batch_convert::<FsG1, FsFp, FsG1Affine>(secret_g1);
                    let points = unsafe {
                        alloc::slice::from_raw_parts(
                            points.as_ptr() as *const blst_p1_affine,
                            points.len(),
                        )
                    };
                    let prepared = rust_kzg_blst_sppark::prepare_multi_scalar_mult(points);
                    Some(Arc::new(PrecomputationTable::from_ptr(prepared)))
                }

                #[cfg(not(feature = "sppark"))]
                {
                    precompute(g1_lagrange_brp).ok().flatten().map(Arc::new)
                }
            },
        })
    }

    fn commit_to_poly(&self, p: &PolyData) -> Result<ArkG1, String> {
        if p.coeffs.len() > self.secret_g1.len() {
            return Err(String::from("Polynomial is longer than secret g1"));
        }

        let mut out = ArkG1::default();
        g1_linear_combination(
            &mut out,
            &self.secret_g1,
            &p.coeffs,
            p.coeffs.len(),
            self.get_precomputation(),
        );

        Ok(out)
    }

    fn compute_proof_single(&self, p: &PolyData, x: &ArkFr) -> Result<ArkG1, String> {
        if p.coeffs.is_empty() {
            return Err(String::from("Polynomial must not be empty"));
        }

        // `-(x0^n)`, where `n` is `1`
        let divisor_0 = x.negate();

        // Calculate `q = p / (x^n - x0^n)` for our reduced case (see `compute_proof_multi` for
        // generic implementation)
        let mut out_coeffs = Vec::from(&p.coeffs[1..]);
        for i in (1..out_coeffs.len()).rev() {
            let tmp = out_coeffs[i].mul(&divisor_0);
            out_coeffs[i - 1] = out_coeffs[i - 1].sub(&tmp);
        }

        let q = PolyData { coeffs: out_coeffs };
        let ret = self.commit_to_poly(&q)?;
        Ok(ret)
        // Ok(compute_single(p, x, self))
    }

    fn check_proof_single(
        &self,
        com: &ArkG1,
        proof: &ArkG1,
        x: &ArkFr,
        y: &ArkFr,
    ) -> Result<bool, String> {
        let x_g2: ArkG2 = G2_GENERATOR.mul(x);
        let s_minus_x: ArkG2 = self.secret_g2[1].sub(&x_g2);
        let y_g1 = G1_GENERATOR.mul(y);
        let commitment_minus_y: ArkG1 = com.sub(&y_g1);

        Ok(pairings_verify(
            &commitment_minus_y,
            &G2_GENERATOR,
            proof,
            &s_minus_x,
        ))
    }

    fn compute_proof_multi(&self, p: &PolyData, x0: &ArkFr, n: usize) -> Result<ArkG1, String> {
        if p.coeffs.is_empty() {
            return Err(String::from("Polynomial must not be empty"));
        }

        if !n.is_power_of_two() {
            return Err(String::from("n must be a power of two"));
        }

        // Construct x^n - x0^n = (x - x0.w^0)(x - x0.w^1)...(x - x0.w^(n-1))
        let mut divisor = PolyData {
            coeffs: Vec::with_capacity(n + 1),
        };

        // -(x0^n)
        let x_pow_n = x0.pow(n);

        divisor.coeffs.push(x_pow_n.negate());

        // Zeros
        for _ in 1..n {
            divisor.coeffs.push(kzg::Fr::zero());
        }

        // x^n
        divisor.coeffs.push(kzg::Fr::one());

        let mut new_polina = p.clone();

        // Calculate q = p / (x^n - x0^n)
        // let q = p.div(&divisor).unwrap();
        let q = new_polina.div(&divisor)?;

        let ret = self.commit_to_poly(&q)?;

        Ok(ret)
    }

    fn check_proof_multi(
        &self,
        com: &ArkG1,
        proof: &ArkG1,
        x: &ArkFr,
        ys: &[ArkFr],
        n: usize,
    ) -> Result<bool, String> {
        if !n.is_power_of_two() {
            return Err(String::from("n is not a power of two"));
        }

        // Interpolate at a coset.
        let mut interp = PolyData {
            coeffs: self.fs.fft_fr(ys, true)?,
        };

        let inv_x = x.inverse(); // Not euclidean?
        let mut inv_x_pow = inv_x;
        for i in 1..n {
            interp.coeffs[i] = interp.coeffs[i].mul(&inv_x_pow);
            inv_x_pow = inv_x_pow.mul(&inv_x);
        }

        // [x^n]_2
        let x_pow = inv_x_pow.inverse();

        let xn2 = G2_GENERATOR.mul(&x_pow);

        // [s^n - x^n]_2
        let xn_minus_yn = self.secret_g2[n].sub(&xn2);

        // [interpolation_polynomial(s)]_1
        let is1 = self.commit_to_poly(&interp).unwrap();

        // [commitment - interpolation_polynomial(s)]_1 = [commit]_1 - [interpolation_polynomial(s)]_1
        let commit_minus_interp = com.sub(&is1);

        let ret = pairings_verify(&commit_minus_interp, &G2_GENERATOR, proof, &xn_minus_yn);

        Ok(ret)
    }

    fn get_roots_of_unity_at(&self, i: usize) -> ArkFr {
        self.fs.get_roots_of_unity_at(i)
    }

    fn get_fft_settings(&self) -> &LFFTSettings {
        &self.fs
    }

    fn get_g1_monomial(&self) -> &[ArkG1] {
        // todo
    }

    fn get_g1_lagrange_brp(&self) -> &[ArkG1] {
        // todo
    }

    fn get_g2_monomial(&self) -> &[ArkG2] {
        // todo
    }

    fn get_x_ext_fft_column(&self, index: usize) -> &[ArkG1] {
        // todo
    }

    fn get_precomputation(&self) -> Option<&PrecomputationTable<ArkFr, ArkG1, ArkFp, ArkG1Affine>> {
        self.precomputation.as_ref().map(|v| v.as_ref())
    }
}

type ArkFpInt = <ark_bls12_381::g1::Config as CurveConfig>::BaseField;
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct ArkFp(pub blst_fp);
impl G1Fp for ArkFp {
    fn one() -> Self {
        Self(blst_fp {
            l: [
                8505329371266088957,
                17002214543764226050,
                6865905132761471162,
                8632934651105793861,
                6631298214892334189,
                1582556514881692819,
            ],
        })
    }
    fn zero() -> Self {
        Self(blst_fp {
            l: [0, 0, 0, 0, 0, 0],
        })
    }
    fn bls12_381_rx_p() -> Self {
        Self(blst_fp {
            l: [
                8505329371266088957,
                17002214543764226050,
                6865905132761471162,
                8632934651105793861,
                6631298214892334189,
                1582556514881692819,
            ],
        })
    }

    fn inverse(&self) -> Option<Self> {
        let mut out: Self = *self;
        unsafe {
            blst::blst_fp_inverse(&mut out.0, &self.0);
        }
        Some(out)
    }

    fn square(&self) -> Self {
        let mut out: Self = Default::default();
        unsafe {
            blst::blst_fp_sqr(&mut out.0, &self.0);
        }
        out
    }

    fn double(&self) -> Self {
        let mut out: Self = Default::default();
        unsafe {
            blst::blst_fp_add(&mut out.0, &self.0, &self.0);
        }
        out
    }

    fn from_underlying_arr(arr: &[u64; 6]) -> Self {
        Self(blst_fp { l: *arr })
    }

    fn neg_assign(&mut self) {
        unsafe {
            blst::blst_fp_cneg(&mut self.0, &self.0, true);
        }
    }

    fn mul_assign_fp(&mut self, b: &Self) {
        unsafe {
            blst::blst_fp_mul(&mut self.0, &self.0, &b.0);
        }
    }

    fn sub_assign_fp(&mut self, b: &Self) {
        unsafe {
            blst::blst_fp_sub(&mut self.0, &self.0, &b.0);
        }
    }

    fn add_assign_fp(&mut self, b: &Self) {
        unsafe {
            blst::blst_fp_add(&mut self.0, &self.0, &b.0);
        }
    }
}

pub struct ArkG1ProjAddAffine;
impl G1ProjAddAffine<ArkG1, ArkFp, ArkG1Affine> for ArkG1ProjAddAffine {
    fn add_assign_affine(proj: &mut ArkG1, aff: &ArkG1Affine) {
        unsafe {
            blst::blst_p1_add_affine(&mut proj.0, &proj.0, &aff.0);
        }
    }

    fn add_or_double_assign_affine(proj: &mut ArkG1, aff: &ArkG1Affine) {
        unsafe {
            blst::blst_p1_add_or_double_affine(&mut proj.0, &proj.0, &aff.0);
        }
    }
}
