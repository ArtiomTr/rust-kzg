use super::P1;
use crate::P2;
use mcl_rust::{Fp, Fp2, G1, G2, Fr};
use blst::{blst_fp, blst_fp2, blst_fr, blst_p1, blst_p2};

#[derive(Debug, PartialEq, Eq)]
pub struct Error;

pub const fn blst_fr_into_mcl_fr(fr: blst_fr) -> Fr {
    Fr{ d: fr.l }
}
pub const fn mcl_fr_into_blst_fr(scalar: Fr) -> blst_fr {
    blst_fr { l: scalar.d }
}
pub const fn blst_fp2_into_mcl_fq2(fp: &blst_fp2) -> Fp2 {
    let c0 = Fp{ d: fp.fp[0].l };
    let c1 = Fp{ d: fp.fp[1].l };
    Fp2 { d: [c0, c1] }
}

pub const fn blst_p1_into_mcl_g1projective(p1: &P1) -> G1 {
    let x = Fp { d: p1.x.l};
    let y = Fp { d: p1.y.l};
    let z = Fp { d: p1.z.l};
    G1 { x, y, z }
}

pub const fn mcl_g1projective_into_blst_p1(p1: G1) -> blst_p1 {
    let x = blst_fp { l: p1.x.d };
    let y = blst_fp { l: p1.y.d };
    let z = blst_fp { l: p1.z.d };

    blst_p1 { x, y, z }
}

pub const fn blst_p2_into_mcl_g2projective(p2: &P2) -> G2 {
    G2 {
        x: blst_fp2_into_mcl_fq2(&p2.x),
        y: blst_fp2_into_mcl_fq2(&p2.y),
        z: blst_fp2_into_mcl_fq2(&p2.z),
    }
}

pub const fn mcl_g2projective_into_blst_p2(p2: G2) -> blst_p2 {
    let x = blst_fp2 {
        fp: [blst_fp { l: p2.x.d[0].d }, blst_fp { l: p2.x.d[1].d }],
    };

    let y = blst_fp2 {
        fp: [blst_fp { l: p2.y.d[0].d }, blst_fp { l: p2.y.d[1].d }],
    };

    let z = blst_fp2 {
        fp: [blst_fp { l: p2.z.d[0].d }, blst_fp { l: p2.z.d[1].d }],
    };

    blst_p2 { x, y, z }
}

/// Computes `a - (b + borrow)`, returning the result and the new borrow.
#[inline(always)]
pub const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let ret = (a as u128).wrapping_sub((b as u128) + ((borrow >> 63) as u128));
    (ret as u64, (ret >> 64) as u64)
}

/// Computes `a + b + carry`, returning the result and the new carry over.
#[inline(always)]
pub const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + (b as u128) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Computes `a + (b * c) + carry`, returning the result and the new carry over.
#[inline(always)]
pub const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + ((b as u128) * (c as u128)) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}