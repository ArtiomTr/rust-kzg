use core::{
    mem::size_of,
    ptr::{self, null_mut},
};

extern crate alloc;
use alloc::vec;

use blst::{
    blst_fp, blst_p1, blst_p1_add, blst_p1_affine, blst_p1_double, blst_p1s_mult_wbits,
    blst_p1s_mult_wbits_precompute, blst_p1s_to_affine, blst_scalar, blst_scalar_from_fr,
    blst_uint64_from_scalar, byte, limb_t,
};
use kzg::G1Mul;

use crate::{
    bgmw::{EXPONENT_OF_Q_BGMW95, H_BGMW95, Q_RADIX_PIPPENGER_VARIANT},
    types::{fr::FsFr, g1::FsG1, kzg_settings::BGMWPreComputationList},
};

#[cfg(feature = "parallel")]
use {
    crate::mult_pippenger::P1Affines,
    core::sync::atomic::{AtomicUsize, Ordering},
    kzg::G1,
    std::sync::{mpsc, Arc},
};

fn pippenger_window_size(mut npoints: usize) -> usize {
    let mut wbits = 0usize;

    loop {
        npoints >>= 1;
        if npoints == 0 {
            break;
        }
        wbits += 1;
    }

    if wbits > 12 {
        wbits - 3
    } else if wbits > 4 {
        wbits - 2
    } else if wbits != 0 {
        2
    } else {
        1
    }
}

fn is_zero(val: limb_t) -> limb_t {
    (!val & (val.wrapping_sub(1))) >> (limb_t::BITS - 1)
}

/// Window value encoding that utilizes the fact that -P is trivially
/// calculated, which allows to halve the size of pre-computed table,
/// is attributed to A. D. Booth, hence the name of the subroutines...
///
/// TODO: figure out how this function works exactly
fn booth_encode(wval: limb_t, sz: usize) -> limb_t {
    let mask = (0 as limb_t).wrapping_sub(wval >> sz);

    let wval = (wval + 1) >> 1;
    (wval ^ mask).wrapping_sub(mask)
}

#[inline(always)]
unsafe fn vec_zero(ret: *mut limb_t, mut num: usize) {
    num /= size_of::<usize>();

    for i in 0..num {
        *ret.add(i) = 0;
    }
}

/// Extract `bits` from the beginning of `d` array, with offset `off`.
///
/// This function is used to extract N bits from the scalar, decomposing it into q-ary representation.
/// This works because `q` is `2^bits`, so extracting `bits` from scalar will break it to correct representation.
///
/// Caution! This function guarantees only that `bits` bits from the right will contain extracted. All unused bits
/// will contain "trash". For example, if we try to extract first 4 bits from the array `[0b01010111u8]`, this
/// function will return `0111`, but other bits will contain trash:
///
/// ```rust
/// let val = rust_kzg_blst::msm::get_wval_limb(&[0b01010111u8], 0, 4);
/// assert_eq!(val, 0b01010111);
/// // if you want to get value containing only extracted bits and zeros, do bitwise and on return value with mask:
/// assert_eq!(val & 0b00001111, 0b00000111);
/// ```
///
/// # Arguments
///
/// * `d`    - byte array, from which bits will be extracted
/// * `off`  - index of first bit, that will be extracted
/// * `bits` - number of bits to extract (up to 25)
///
/// # Example
///
/// Scalars are represented with 32 bytes. To simplify example, let's say our scalars are only 4 bytes long.
/// Then, we can take `q` as `2^6`. Then consider scalar value `4244836224`, bytes: `[128u8, 15u8, 3u8, 253u8]`
/// (little-endian order). So if we repeatedly take 6 bits from this scalar, we will get q-ary representation
/// of this scalar:
///
/// ```rust
/// let scalar = [0b10000000, 0b00001111u8, 0b00000011u8, 0b11111101u8]; // this is [128u8, 15u8, 3u8, 253u8] written in binary
/// let limb_1 = rust_kzg_blst::msm::get_wval_limb(&scalar, 0, 6);
/// // function leaves trash on all other bytes, so real answer only in 6 bits from right
/// assert_eq!(limb_1 & 0b00111111, 0b00000000 /*  0 */); // 11111101000000110000111110|000000|
/// let limb_2 = rust_kzg_blst::msm::get_wval_limb(&scalar, 6, 6);
/// assert_eq!(limb_2 & 0b00111111, 0b00111110 /* 62 */); // 11111101000000110000|111110|000000
/// let limb_3 = rust_kzg_blst::msm::get_wval_limb(&scalar, 12, 6);
/// assert_eq!(limb_3 & 0b00111111, 0b00110000 /* 48 */); // 11111101000000|110000|111110000000
/// let limb_4 = rust_kzg_blst::msm::get_wval_limb(&scalar, 18, 6);
/// assert_eq!(limb_4 & 0b00111111, 0b00000000 /*  0 */); // 11111101|000000|110000111110000000
/// let limb_5 = rust_kzg_blst::msm::get_wval_limb(&scalar, 24, 6);
/// assert_eq!(limb_5 & 0b00111111, 0b00111101 /* 61 */); // 11|111101|000000110000111110000000
/// let limb_r = rust_kzg_blst::msm::get_wval_limb(&scalar, 28, 8 % 6); // get remaining part
/// assert_eq!(limb_r & 0b00000011, 0b00000011 /*  3 */); // |11|111101000000110000111110000000
/// ```
///
/// And that gives q-ary representation of scalar `4244836224`, where `q` = `2^6` = `64`:
///
/// 4244836224 = 0 * 64^0 + 62 * 64^1 + 48 * 64^2 + 0 * 64^3 + 61 * 64^4 + 3 * 64^5
///
pub fn get_wval_limb(mut d: &[u8], off: usize, bits: usize) -> limb_t {
    // Calculate topmost byte that needs to be considered.
    let top = ((off + bits - 1) / 8).wrapping_sub((off / 8).wrapping_sub(1));

    // Skipping first `off/8` of bytes, because offset specified how many bits must be ignored
    d = &d[off / 8..];

    // For first iteration, none bits will be ignored - all bits added to result
    let mut mask = limb_t::MAX;

    let mut ret: limb_t = 0;
    for i in 0..4usize {
        /*
         * Add bits from current byte to the result.
         *
         * Applying bitwise and (&) on current byte and mask will keep or ignore all bits from current byte, because
         * mask can only be 0 or limb_t::MAX. Doing right bit shift will move those bits to correct position, e.g. when
         * `i=0` (we are processing first byte), bits won't move, when `i=1` bits will be moved by 8 (1 byte) and so on.
         * After that, we will get value, that is zero-padded from the right and left, so doing bitwise or (|) operation
         * with the result, will just append bytes to it.
         */
        ret |= (d[0] as limb_t & mask) << (8 * i);

        /*
         * Create new mask - either 0 or limb_t::MAX.
         *
         * If `i+1` is greater than or equal to `top`, then byte must be ignored, so the mask is set to `0`. Otherwise,
         * mask is set to `limb_t::MAX` (include all bits). This is done for branch optimization (avoid if).
         */
        mask =
            (0 as limb_t).wrapping_sub(((i + 1).wrapping_sub(top) >> (usize::BITS - 1)) as limb_t);

        /*
         * Conditionally move current array by `1`, if not all needed bytes already read.
         *
         * This is done by applying bitwise and (&) on `1` and `mask`. Because mask is `0` when `i + 1` is >= `top`,
         * doing bitwise and will result in `0`, so slice will not be moved. Otherwise, mask will be `limb_t::MAX`, and
         * slice will be moved by `1`.
         */
        d = &d[(1 & mask).try_into().unwrap()..];
    }

    // Because offset won't always be divisible by `8`, we need to ignore remaining bits.
    ret >> (off % 8)
}

#[inline(always)]
unsafe fn vec_is_zero(a: *const byte, num: usize) -> limb_t {
    let ap = a as *const limb_t;
    let num = num / size_of::<limb_t>();

    let mut acc: limb_t = 0;
    for i in 0..num {
        acc |= *ap.wrapping_add(i);
    }

    is_zero(acc)
}

#[inline(always)]
unsafe fn vec_copy(ret: *mut u8, a: *const u8, num: usize) {
    let rp = ret as *mut limb_t;
    let ap = a as *const limb_t;

    let num = num / size_of::<limb_t>();

    for i in 0..num {
        *rp.wrapping_add(i) = *ap.wrapping_add(i);
    }
}

const BLS12_381_RX_P: blst_fp = blst_fp {
    l: [
        8505329371266088957,
        17002214543764226050,
        6865905132761471162,
        8632934651105793861,
        6631298214892334189,
        1582556514881692819,
    ],
};

unsafe fn p1_dadd_affine(
    p3: *mut P1XYZZ,
    p1: *const P1XYZZ,
    p2: *const blst_p1_affine,
    subtract: limb_t,
) {
    // POINTonE1xyzz *p3, const POINTonE1xyzz *p1, const POINTonE1_affine *p2, bool_t subtract
    // vec384 P, R;

    // if (vec_is_zero(p2, sizeof(*p2)))
    if vec_is_zero(p2 as *const u8, size_of::<blst_p1_affine>()) != 0 {
        // vec_copy(p3, p1, sizeof(*p3));
        vec_copy(p3 as *mut u8, p1 as *const u8, size_of::<P1XYZZ>());
        return;
    // else if (vec_is_zero(p1->ZZZ, 2 * sizeof(p1->ZZZ)))
    } else if vec_is_zero(
        &(*p1).zzz as *const blst_fp as *const u8,
        2 * size_of::<blst_fp>(),
    ) != 0
    {
        // vec_copy(p3->X, p2->X, 2 * sizeof(p3->X));
        vec_copy(
            &mut ((*p3).x) as *mut blst_fp as *mut u8,
            &((*p2).x) as *const blst_fp as *const u8,
            2 * size_of::<blst_fp>(),
        );
        // cneg_fp(p3->ZZZ, BLS12_381_Rx.p, subtract);
        blst::blst_fp_cneg(&mut (*p3).zzz, &BLS12_381_RX_P, subtract != 0);
        // vec_copy(p3->ZZ, BLS12_381_Rx.p, sizeof(p3->ZZ));
        vec_copy(
            &mut ((*p3).zz) as *mut blst_fp as *mut u8,
            &BLS12_381_RX_P as *const blst_fp as *const u8,
            size_of::<blst_fp>(),
        );
        // return
        return;
    }

    let mut p = blst_fp::default();
    let mut r = blst_fp::default();

    // mul_fp(P, p2->X, p1->ZZ);
    blst::blst_fp_mul(&mut p, &(*p2).x, &(*p1).zz);
    // mul_fp(R, p2->Y, p1->ZZZ);
    blst::blst_fp_mul(&mut r, &(*p2).y, &(*p1).zzz);
    // cneg_fp(R, R, subtract);
    blst::blst_fp_cneg(&mut r, &r, subtract != 0);
    // sub_fp(P, P, p1->X);
    blst::blst_fp_sub(&mut p, &p, &(*p1).x);
    // sub_fp(R, R, p1->Y);
    blst::blst_fp_sub(&mut r, &r, &(*p1).y);
    // if (!vec_is_zero(P, sizeof(P)))
    if vec_is_zero(&p as *const blst_fp as *const u8, size_of::<blst_fp>()) == 0 {
        // vec384 PP, PPP, Q;
        let mut pp = blst_fp::default();
        let mut ppp = blst_fp::default();
        let mut q = blst_fp::default();
        // sqr_fp(PP, P);
        blst::blst_fp_sqr(&mut pp, &p);
        // mul_fp(PPP, PP, P);
        blst::blst_fp_mul(&mut ppp, &pp, &p);
        // mul_fp(Q, p1->X, PP);
        blst::blst_fp_mul(&mut q, &(*p1).x, &pp);
        // sqr_fp(p3->X, R);
        blst::blst_fp_sqr(&mut (*p3).x, &r);
        // add_fp(P, Q, Q);
        blst::blst_fp_add(&mut p, &q, &q);
        // sub_fp(p3->X, p3->X, PPP);
        blst::blst_fp_sub(&mut (*p3).x, &(*p3).x, &ppp);
        // sub_fp(p3->X, p3->X, P);
        blst::blst_fp_sub(&mut (*p3).x, &(*p3).x, &p);
        // sub_fp(Q, Q, p3->X);
        blst::blst_fp_sub(&mut q, &q, &(*p3).x);
        // mul_fp(Q, Q, R);
        blst::blst_fp_mul(&mut q, &q, &r);
        // mul_fp(p3->Y, p1->Y, PPP);
        blst::blst_fp_mul(&mut (*p3).y, &(*p1).y, &ppp);
        // sub_fp(p3->Y, Q, p3->Y);
        blst::blst_fp_sub(&mut (*p3).y, &q, &(*p3).y);
        // mul_fp(p3->ZZ, p1->ZZ, PP);
        blst::blst_fp_mul(&mut (*p3).zz, &(*p1).zz, &pp);
        // mul_fp(p3->ZZZ, p1->ZZZ, PPP);
        blst::blst_fp_mul(&mut (*p3).zzz, &(*p1).zzz, &ppp);
    // else if (vec_is_zero(R, sizeof(R)))
    } else if vec_is_zero(&r as *const blst_fp as *const u8, size_of::<blst_fp>()) != 0 {
        // vec384 U, S, M;
        let mut u = blst_fp::default();
        let mut s = blst_fp::default();
        let mut m = blst_fp::default();
        // add_fp(U, p2->Y, p2->Y);
        blst::blst_fp_add(&mut u, &(*p2).y, &(*p2).y);
        // sqr_fp(p3->ZZ, U);
        blst::blst_fp_sqr(&mut (*p3).zz, &u);
        // mul_fp(p3->ZZZ, p3->ZZ, U);
        blst::blst_fp_mul(&mut (*p3).zzz, &(*p3).zz, &u);
        // mul_fp(S, p2->X, p3->ZZ);
        blst::blst_fp_mul(&mut s, &(*p2).x, &(*p3).zz);
        // sqr_fp(M, p2->X);
        blst::blst_fp_sqr(&mut m, &(*p2).x);
        // mul_by_3_fp(M, M);
        blst::blst_fp_mul_by_3(&mut m, &m);
        // sqr_fp(p3->X, M);
        blst::blst_fp_sqr(&mut (*p3).x, &m);
        // add_fp(U, S, S);
        blst::blst_fp_add(&mut u, &s, &s);
        // sub_fp(p3->X, p3->X, U);
        blst::blst_fp_sub(&mut (*p3).x, &(*p3).x, &u);
        // mul_fp(p3->Y, p3->ZZZ, p2->Y);
        blst::blst_fp_mul(&mut (*p3).y, &(*p3).zzz, &(*p2).y);
        // sub_fp(S, S, p3->X);
        blst::blst_fp_sub(&mut s, &s, &(*p3).x);
        // mul_fp(S, S, M);
        blst::blst_fp_mul(&mut s, &s, &m);
        // sub_fp(p3->Y, S, p3->Y);
        blst::blst_fp_sub(&mut (*p3).y, &s, &(*p3).y);
        // cneg_fp(p3->ZZZ, p3->ZZZ, subtract);
        blst::blst_fp_cneg(&mut (*p3).zzz, &(*p3).zzz, subtract != 0);
    } else {
        // vec_zero(p3->ZZZ, 2 * sizeof(p3->ZZZ));
        vec_zero(
            &mut (*p3).zzz as *mut blst_fp as *mut u64,
            2 * size_of::<blst_fp>(),
        );
    }
}

unsafe fn p1_dadd(p3: *mut P1XYZZ, p1: *const P1XYZZ, p2: *const P1XYZZ) {
    // POINTonE1xyzz *p3, const POINTonE1xyzz *p1, const POINTonE1xyzz *p2

    // vec384 U, S, P, R;

    // if (vec_is_zero(p2->ZZZ, 2 * sizeof(p2->ZZZ)))
    if vec_is_zero(
        &(*p2).zzz as *const blst_fp as *const u8,
        2 * size_of::<blst_fp>(),
    ) != 0
    {
        // vec_copy(p3, p1, sizeof(*p3));
        vec_copy(p3 as *mut u8, p1 as *const u8, size_of::<P1XYZZ>());
        // return;
        return;
    // else if (vec_is_zero(p1->ZZZ, 2 * sizeof(p1->ZZZ)))
    } else if vec_is_zero(
        &(*p1).zzz as *const blst_fp as *const u8,
        2 * size_of::<blst_fp>(),
    ) != 0
    {
        // vec_copy(p3, p2, sizeof(*p3));
        vec_copy(p3 as *mut u8, p2 as *mut u8, size_of::<P1XYZZ>());
        // return;
        return;
    }

    let mut u = blst_fp::default();
    let mut s = blst_fp::default();
    let mut p = blst_fp::default();
    let mut r = blst_fp::default();

    // mul_fp(U, p1->X, p2->ZZ);
    blst::blst_fp_mul(&mut u, &(*p1).x, &(*p2).zz);
    // mul_fp(S, p1->Y, p2->ZZZ);
    blst::blst_fp_mul(&mut s, &(*p1).y, &(*p2).zzz);
    // mul_fp(P, p2->X, p1->ZZ);
    blst::blst_fp_mul(&mut p, &(*p2).x, &(*p1).zz);
    // mul_fp(R, p2->Y, p1->ZZZ);
    blst::blst_fp_mul(&mut r, &(*p2).y, &(*p1).zzz);
    // sub_fp(P, P, U);
    blst::blst_fp_sub(&mut p, &p, &u);
    // sub_fp(R, R, S);
    blst::blst_fp_sub(&mut r, &r, &s);

    // if (!vec_is_zero(P, sizeof(P)))
    if vec_is_zero(&p as *const blst_fp as *const u8, size_of::<blst_fp>()) == 0 {
        // vec384 PP, PPP, Q;
        let mut pp = blst_fp::default();
        let mut ppp = blst_fp::default();
        let mut q = blst_fp::default();
        // sqr_fp(PP, P);
        blst::blst_fp_sqr(&mut pp, &p);
        // mul_fp(PPP, PP, P);
        blst::blst_fp_mul(&mut ppp, &pp, &p);
        // mul_fp(Q, U, PP);
        blst::blst_fp_mul(&mut q, &u, &pp);
        // sqr_fp(p3->X, R);
        blst::blst_fp_sqr(&mut (*p3).x, &r);
        // add_fp(P, Q, Q);
        blst::blst_fp_add(&mut p, &q, &q);
        // sub_fp(p3->X, p3->X, PPP);
        blst::blst_fp_sub(&mut (*p3).x, &(*p3).x, &ppp);
        // sub_fp(p3->X, p3->X, P);
        blst::blst_fp_sub(&mut (*p3).x, &(*p3).x, &p);
        // sub_fp(Q, Q, p3->X);
        blst::blst_fp_sub(&mut q, &q, &(*p3).x);
        // mul_fp(Q, Q, R);
        blst::blst_fp_mul(&mut q, &q, &r);
        // mul_fp(p3->Y, S, PPP);
        blst::blst_fp_mul(&mut (*p3).y, &s, &ppp);
        // sub_fp(p3->Y, Q, p3->Y);
        blst::blst_fp_sub(&mut (*p3).y, &q, &(*p3).y);
        // mul_fp(p3->ZZ, p1->ZZ, p2->ZZ);
        blst::blst_fp_mul(&mut (*p3).zz, &(*p1).zz, &(*p2).zz);
        // mul_fp(p3->ZZZ, p1->ZZZ, p2->ZZZ);
        blst::blst_fp_mul(&mut (*p3).zzz, &(*p1).zzz, &(*p2).zzz);
        // mul_fp(p3->ZZ, p3->ZZ, PP);
        blst::blst_fp_mul(&mut (*p3).zz, &(*p3).zz, &pp);
        // mul_fp(p3->ZZZ, p3->ZZZ, PPP);
        blst::blst_fp_mul(&mut (*p3).zzz, &(*p3).zzz, &ppp);
    // else if (vec_is_zero(R, sizeof(R)))
    } else if vec_is_zero(&r as *const blst_fp as *const u8, size_of::<blst_fp>()) != 0 {
        // vec384 V, W, M;
        let mut v = blst_fp::default();
        let mut w = blst_fp::default();
        let mut m = blst_fp::default();

        // add_fp(U, p1->Y, p1->Y);
        blst::blst_fp_add(&mut u, &(*p1).y, &(*p1).y);
        // sqr_fp(V, U);
        blst::blst_fp_sqr(&mut v, &u);
        // mul_fp(W, V, U);
        blst::blst_fp_mul(&mut w, &v, &u);
        // mul_fp(S, p1->X, V);
        blst::blst_fp_mul(&mut s, &(*p1).x, &v);
        // sqr_fp(M, p1->X);
        blst::blst_fp_sqr(&mut m, &(*p1).x);
        // mul_by_3_fp(M, M);
        blst::blst_fp_mul_by_3(&mut m, &m);
        // sqr_fp(p3->X, M);
        blst::blst_fp_sqr(&mut (*p3).x, &m);
        // add_fp(U, S, S);
        blst::blst_fp_add(&mut u, &s, &s);
        // sub_fp(p3->X, p3->X, U);
        blst::blst_fp_sub(&mut (*p3).x, &(*p3).x, &u);
        // mul_fp(p3->Y, W, p1->Y);
        blst::blst_fp_mul(&mut (*p3).y, &w, &(*p1).y);
        // sub_fp(S, S, p3->X);
        blst::blst_fp_sub(&mut s, &s, &(*p3).x);
        // mul_fp(S, S, M);
        blst::blst_fp_mul(&mut s, &s, &m);
        // sub_fp(p3->Y, S, p3->Y);
        blst::blst_fp_sub(&mut (*p3).y, &s, &(*p3).y);
        // mul_fp(p3->ZZ, p1->ZZ, V);
        blst::blst_fp_mul(&mut (*p3).zz, &(*p1).zz, &v);
        // mul_fp(p3->ZZZ, p1->ZZZ, W);
        blst::blst_fp_mul(&mut (*p3).zzz, &(*p1).zzz, &w);
    } else {
        // vec_zero(p3->ZZZ, 2 * sizeof(p3->ZZZ));
        vec_zero(
            &mut (*p3).zzz as *mut blst_fp as *mut limb_t,
            2 * size_of::<blst_fp>(),
        );
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
struct P1XYZZ {
    x: blst_fp,
    y: blst_fp,
    zzz: blst_fp,
    zz: blst_fp,
}

/// Move point to corresponding bucket
///
/// This method will decode `booth_idx`, and add or subtract point to bucket.
/// booth_idx contains bucket index and sign. Sign shows, if point needs to be added to or subtracted from bucket.
///
/// ## Arguments:
///
/// * buckets   - pointer to the bucket array beginning
/// * booth_idx - bucket index, encoded with [booth_encode] function
/// * wbits     - window size, aka exponent of q (q^window)
/// * point     - point to move
///
unsafe fn p1s_bucket(
    buckets: *mut P1XYZZ,
    mut booth_idx: limb_t,
    wbits: usize,
    point: *const blst_p1_affine,
) {
    /*
     * Get the `wbits + 1` bit in booth index.
     * This is a sign bit: `0` means scalar is positive, `1` - negative
     */
    let booth_sign = (booth_idx >> wbits) & 1;

    /*
     * Normalize bucket index.
     *
     * `(1 << wbits) - 1` generates number, which has `wbits` ones at the end.
     * For example:
     *     `wbits = 3` -> 0b00000111 (7)
     *     `wbits = 4` -> 0b00001111 (15)
     *     `wbits = 5` -> 0b00011111 (31)
     * And so on.
     *
     * Applying bitwise and (&) on `booth_idx` with such mask, means "leave only `wbits` bits from the end, and set all others to zero"
     * For example:
     *     `booth_idx = 14`,  `wbits = 3` -> 0b00001110 & 0b00000111 = 0b00000110
     *     `booth_idx = 255`, `wbits = 4` -> 0b11111111 & 0b00001111 = 0b00001111
     *     `booth_idx = 253`, `wbits = 5` -> 0b11111101 & 0b00011111 = 0b00011101
     */
    booth_idx &= (1 << wbits) - 1;

    // Bucket with index zero is ignored, as all values that fall in it are multiplied by zero (P * 0 = 0, no need to compute that).
    if booth_idx != 0 {
        // This command moves all buckets to the right, as bucket 0 doesn't exist (P * 0 = 0, no need to save it).
        booth_idx -= 1;

        /*
         * When:
         *     `booth_sign = 0` -> add point to bucket[booth_idx]
         *     `booth_sign = 1` -> subtract point from bucket[booth_idx]
         */
        p1_dadd_affine(
            buckets.wrapping_add(booth_idx.try_into().unwrap()),
            buckets.wrapping_add(booth_idx.try_into().unwrap()),
            point,
            booth_sign,
        );
    }
}

unsafe fn p1_to_jacobian(out: *mut blst_p1, input: *const P1XYZZ) {
    // POINTonE1 *out, const POINTonE1xyzz *in

    // blst::blst_p1_from_jacobian(out, in_)

    // mul_fp(out->X, in->X, in->ZZ);
    blst::blst_fp_mul(&mut (*out).x, &(*input).x, &(*input).zz);
    // mul_fp(out->Y, in->Y, in->ZZZ);
    blst::blst_fp_mul(&mut (*out).y, &(*input).y, &(*input).zzz);
    // vec_copy(out->Z, in->ZZ, sizeof(out->Z));
    vec_copy(
        &mut (*out).z as *mut blst_fp as *mut u8,
        &(*input).zz as *const blst_fp as *const u8,
        size_of::<blst_fp>(),
    );
}

/// Calculate bucket sum
///
/// This function multiplies point in each bucket by it's index. Then, it will sum all multiplication results and write
/// resulting point to the `out`.
///
/// This function also clears all buckets (sets all values in buckets to zero.)
///
/// ## Arguments
///
/// * out     - output where bucket sum must be written
/// * buckets - pointer to the beginning of the array of buckets
/// * wbits   - window size, aka exponent of q (q^window)
///  
unsafe fn p1_integrate_buckets(out: *mut blst_p1, buckets: *mut P1XYZZ, wbits: usize) {
    // Resulting point
    let mut ret = [P1XYZZ::default()];
    // Accumulator
    let mut acc = [P1XYZZ::default()];

    // Calculate total amount of buckets
    let mut n = (1usize << wbits) - 1;

    // Copy last point to the accumulator
    vec_copy(
        &mut acc as *mut P1XYZZ as *mut u8,
        buckets.wrapping_add(n) as *const u8,
        size_of::<P1XYZZ>(),
    );

    // Copy last point to the return value
    vec_copy(
        &mut ret as *mut P1XYZZ as *mut u8,
        buckets.wrapping_add(n) as *const u8,
        size_of::<P1XYZZ>(),
    );

    // Clear last bucket
    vec_zero(buckets.wrapping_add(n) as *mut limb_t, size_of::<P1XYZZ>());

    /*
     * Sum all buckets.
     *
     * Starting from the end, this loop adds point to accumulator, and then adds point to the result.
     * If the point is in the bucket `i`, then adding this point to the accumulator and adding accumulator `i` times
     * helps to avoid multiplication of point by `i`.
     *
     * Example:
     *
     * If we have 3 buckets with points [`p1`, `p2`, `p3`], and we need to calculate bucket sum, naive approach would be:
     * `S` = `p1` + 2 * `p2` + 3 * `p3` (which is `p1` + `p2` + `p2` + `p3` + `p3` + `p3` - 5 additions)
     * But using accumulator, it would be:
     *
     * ```rust
     * acc = p3;
     * ret = p3;
     * acc += p2;   // now acc contains `p2` + `p3`
     * ret += acc;  // now res contains `p2` + 2*`p3`
     * acc += p1;   // now acc contains `p1` + `p2` + `p3`
     * ret += acc;  // now res contains `p1` + 2*`p2` + 3*`p3`
     * ```
     *
     * 4 additions. So using accumulator, we saved 1 addition.
     */
    loop {
        if n == 0 {
            break;
        }
        n -= 1;

        // Add point to accumulator
        p1_dadd(acc.as_mut_ptr(), acc.as_ptr(), buckets.wrapping_add(n));
        // Add accumulator to result
        p1_dadd(ret.as_mut_ptr(), ret.as_ptr(), acc.as_ptr());
        // Clear bucket
        vec_zero(buckets.wrapping_add(n) as *mut limb_t, size_of::<P1XYZZ>());
    }

    // Convert point from magical 4-coordinate system to Jacobian (normal)
    p1_to_jacobian(out, ret.as_ptr());
}

// unsafe fn p1_prefetch() {
// booth_idx &= (1 << wbits) - 1;
// if (booth_idx--)
//     vec_prefetch(&buckets[booth_idx], sizeof(buckets[booth_idx]));
// }

#[allow(clippy::too_many_arguments)]
pub unsafe fn p1s_tile_pippenger_pub(
    ret: *mut blst_p1,
    points: &[blst_p1_affine],
    npoints: usize,
    scalars: &[u8],
    nbits: usize,
    buckets: *mut limb_t,
    bit0: usize,
    window: usize,
) {
    let (wbits, cbits) = if bit0 + window > nbits {
        let wbits = nbits - bit0;
        (wbits, wbits + 1)
    } else {
        (window, window)
    };

    p1s_tile_pippenger(
        ret, points, npoints, scalars, nbits, buckets, bit0, wbits, cbits,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn p1s_tile_pippenger(
    ret: *mut blst_p1,
    points: &[blst_p1_affine],
    mut npoints: usize,
    scalars: &[u8],
    nbits: usize,
    buckets: *mut limb_t,
    bit0: usize,
    wbits: usize,
    cbits: usize,
) {
    // Calculate number of bytes, to fit `nbits`. Basically, this is division by 8 with rounding up to nearest integer.
    let nbytes = (nbits + 7) / 8;

    // Get first scalar
    let scalar = &scalars[0..nbytes];

    // Get first point
    let point = &points[0];

    // Create mask, that contains `wbits` ones at the end.
    let wmask = ((1 as limb_t) << (wbits + 1)) - 1;

    /*
     * Check if `bit0` is zero. `z` is set to `1` when `bit0 = 0`, and `0` otherwise.
     *
     * The `z` flag is used to do a small trick -
     */
    let z = is_zero(bit0.try_into().unwrap());

    // Offset `bit0` by 1, if it is not equal to zero.
    let bit0 = bit0 - (z ^ 1) as usize;

    // Increase `wbits` by one, if `bit0` is not equal to zero.
    let wbits = wbits + (z ^ 1) as usize;

    // Calculate first window value (encoded bucket index)
    let wval = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let mut wval = booth_encode(wval, cbits);

    // Get second scalar
    let scalar = &scalars[nbytes..2 * nbytes];

    // Calculate second window value (encoded bucket index)
    let wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let mut wnxt = booth_encode(wnxt, cbits);

    // Move first point to corresponding bucket
    p1s_bucket(buckets as *mut P1XYZZ, wval, cbits, point);

    // Last point will be calculated separately, so decrementing point count
    npoints -= 1;

    // Move points to buckets
    for i in 1..npoints {
        // Get current window value (encoded bucket index)
        wval = wnxt;

        // Get next scalar
        let scalar = &scalars[(i + 1) * nbytes..(i + 2) * nbytes];
        // Get next window value (encoded bucket index)
        wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
        wnxt = booth_encode(wnxt, cbits);

        // TODO: add prefetching
        // POINTonE1_prefetch(buckets, wnxt, cbits);
        // p1_prefetch(buckets, wnxt, cbits);

        // Get current point
        let point = &points[i];

        // Move point to corresponding bucket (add or subtract from bucket)
        // `wval` contains encoded bucket index, as well as sign, which shows if point should be subtracted or added to bucket
        p1s_bucket(buckets as *mut P1XYZZ, wval, cbits, point);
    }
    // Get last point
    let point = &points[npoints];
    // Move point to bucket
    p1s_bucket(buckets as *mut P1XYZZ, wnxt, cbits, point);
    // Integrate buckets - multiply point in each bucket by scalar and sum all results
    p1_integrate_buckets(ret, buckets as *mut P1XYZZ, cbits - 1);
}

pub unsafe fn pippenger(
    ret: *mut blst_p1,
    points: &[blst_p1_affine],
    npoints: usize,
    scalars: &[u8],
    nbits: usize,
    buckets: *mut limb_t,
    window: usize,
) {
    // Calculate exponent of q, if not specified
    let window = if window != 0 {
        window
    } else {
        pippenger_window_size(npoints)
    };

    // Clear buckets (set all to zeros)
    vec_zero(buckets, size_of::<limb_t>() << (window - 1));
    // Clear return value
    vec_zero(ret as *mut limb_t, size_of::<blst_p1>());

    let mut wbits: usize = nbits % window;
    let mut cbits: usize = wbits + 1;
    let mut bit0: usize = nbits;
    let mut tile = [blst_p1::default(); 1];

    loop {
        bit0 -= wbits;
        if bit0 == 0 {
            break;
        }

        p1s_tile_pippenger(
            tile.as_mut_ptr(),
            points,
            npoints,
            scalars,
            nbits,
            buckets,
            bit0,
            wbits,
            cbits,
        );

        // add bucket sum (aka tile) to the return value
        blst_p1_add(ret, ret, tile.as_mut_ptr());
        // multiply return value by Q (2^`window`) - double point `window` times.
        for _ in 0..window {
            blst_p1_double(ret, ret);
        }
        cbits = window;
        wbits = window;
    }
    p1s_tile_pippenger(
        tile.as_mut_ptr(),
        points,
        npoints,
        scalars,
        nbits,
        buckets,
        0,
        wbits,
        cbits,
    );
    blst_p1_add(ret, ret, tile.as_mut_ptr());

    // // vec_zero(buckets, sizeof(buckets[0]) << (window - 1));
    // // vec_zero(ret, sizeof(*ret));
    // // wbits = nbits % window;
    // // cbits = wbits + 1;
    // // while (bit0 -= wbits)
    // // {
    // //     POINTonE1s_tile_pippenger(tile, points, npoints, scalars, nbits, buckets, bit0, wbits, cbits);
    // //     POINTonE1_dadd(ret, ret, tile, NULL);
    // //     for (i = 0; i < window; i++)
    // //         POINTonE1_double(ret, ret);
    // //     cbits = wbits = window;
    // // }
    // // POINTonE1s_tile_pippenger(tile, points, npoints, scalars, nbits, buckets, 0, wbits, cbits);
    // // POINTonE1_dadd(ret, ret, tile, NULL);
}

// static void POINTonE1_bucket_CHES(POINTonE1xyzz buckets[], limb_t booth_idx, const POINTonE1_affine *p, unsigned char booth_sign) { POINTonE1xyzz_dadd_affine(&buckets[booth_idx], &buckets[booth_idx], p, booth_sign); }

fn p1s_bucket_ches(
    buckets: &mut [P1XYZZ],
    booth_idx: limb_t,
    point: &blst_p1_affine,
    booth_sign: u8,
) {
    unsafe {
        p1_dadd_affine(
            buckets
                .as_mut_ptr()
                .wrapping_add(booth_idx.try_into().unwrap()),
            buckets
                .as_mut_ptr()
                .wrapping_add(booth_idx.try_into().unwrap()),
            point,
            booth_sign.into(),
        );
    }
}

fn bgmw_pippenger_tile(
    ret: &mut FsG1,
    points: &[blst_p1_affine],
    npoints: usize,
    scalars: &[i32],
    booth_signs: &[u8],
    buckets: &mut [P1XYZZ],
    q_exponent: usize,
) {
    // POINTonE1 *ret, const POINTonE1_affine *const points[], size_t npoints, const int scalars[], const unsigned char booth_signs[], POINTonE1xyzz buckets[], size_t q_exponent

    // size_t bucket_set_size = (size_t)(1 << (q_exponent - 1)) + 1;
    //// let bucket_set_size = (1usize << (q_exponent - 1)) + 1;
    // vec_zero(buckets, sizeof(buckets[0]) * bucket_set_size);
    //// we don't need to set all buckets to zero, as they are already all zeros
    //// buckets.iter().for_each(|bucket| *bucket = P1XYZZ);
    // vec_zero(ret, sizeof(*ret));
    *ret = FsG1(blst_p1 {
        x: blst_fp { l: [0u64; 6] },
        y: blst_fp { l: [0u64; 6] },
        z: blst_fp { l: [0u64; 6] },
    });
    // int booth_idx, booth_idx_nxt;
    // size_t i;
    // unsigned char booth_sign;

    // const POINTonE1_affine *point = *points++;
    let point = &points[0];
    // booth_idx = *scalars++;
    let booth_idx = scalars[0];
    // booth_sign = *booth_signs++;
    let booth_sign = booth_signs[0];
    // booth_idx_nxt = *scalars++;
    let mut booth_idx_nxt = scalars[1];

    // if (booth_idx)
    if booth_idx != 0 {
        // POINTonE1_bucket_CHES(buckets, booth_idx, point, booth_sign);
        p1s_bucket_ches(buckets, booth_idx as limb_t, point, booth_sign);
    }
    // --npoints;
    let npoints = npoints - 1;
    // for (i = 1; i < npoints; ++i)
    for i in 1..npoints {
        // booth_idx = booth_idx_nxt;
        let booth_idx = booth_idx_nxt;
        // booth_idx_nxt = *scalars++;
        booth_idx_nxt = scalars[i + 1];

        // TODO:
        // POINTonE1_prefetch_CHES(buckets, booth_idx_nxt);

        // point = *points++;
        let point = &points[i];
        // booth_sign = *booth_signs++;
        let booth_sign = booth_signs[i];
        // if (booth_idx)
        if booth_idx != 0 {
            // POINTonE1_bucket_CHES(buckets, booth_idx, point, booth_sign);
            p1s_bucket_ches(buckets, booth_idx as limb_t, point, booth_sign);
        }
    }
    // point = *points;
    let point = &points[npoints];
    // booth_sign = *booth_signs;
    let booth_sign = booth_signs[npoints];
    // POINTonE1_bucket_CHES(buckets, booth_idx_nxt, point, booth_sign);
    p1s_bucket_ches(buckets, booth_idx_nxt as limb_t, point, booth_sign);

    // ++buckets;
    // let buckets = buckets.wrapping_add(1);

    // POINTonE1_integrate_buckets(ret, buckets, q_exponent - 1);
    unsafe {
        p1_integrate_buckets(
            &mut ret.0,
            buckets.as_mut_ptr().wrapping_add(1),
            q_exponent - 1,
        );
    }
}

fn uint256_sbb(a: u64, b: u64, borrow_in: u64) -> (u64, u64) {
    let t_1 = a - (borrow_in >> 63);
    let borrow_temp_1 = t_1 > a;
    let t_2 = t_1 - b;
    let borrow_temp_2 = t_2 > t_1;

    (
        t_2,
        0u64.wrapping_sub((borrow_temp_1 | borrow_temp_2).into()),
    )
}

fn uint256_sbb_discard_hi(a: u64, b: u64, borrow_in: u64) -> u64 {
    a - b - (borrow_in >> 63)
}

fn uint256_sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let (r0, t0) = uint256_sbb(a[0], b[0], 0);
    let (r1, t1) = uint256_sbb(a[1], b[1], t0);
    let (r2, t2) = uint256_sbb(a[2], b[2], t1);
    let r3 = uint256_sbb_discard_hi(a[3], b[3], t2);

    [r0, r1, r2, r3]
}

fn uint256_bithsift(a: &[u64; 4], bits: u64) -> [u64; 4] {
    if bits == 0 {
        return *a;
    }

    let num_shifted_limbs = bits >> 6;
    let limb_shift = bits & 63;

    let mut shifted_limbs = [0u64; 4];
    if limb_shift == 0 {
        shifted_limbs[..4].copy_from_slice(&a[..4]);
    } else {
        let remainder_shift = 64 - limb_shift;
        shifted_limbs[3] = a[3] >> limb_shift;
        let remainder = a[3] << remainder_shift;
        shifted_limbs[2] = (a[2] >> limb_shift) + remainder;
        let remainder = a[2] << remainder_shift;
        shifted_limbs[1] = (a[1] >> limb_shift) + remainder;
        let remainder = a[1] << remainder_shift;
        shifted_limbs[0] = (a[0] >> limb_shift) + remainder;
    };

    let mut result = [0u64; 4];

    for i in 0..((4 - num_shifted_limbs) as usize) {
        result[i] = shifted_limbs[i + (num_shifted_limbs as usize)];
    }

    result
}

fn trans_uint256_to_qhalf_expr(ret_qhalf_expr: &mut [i32], mut a: [u64; 4]) {
    // uint256_t tmp = a;
    // int qhalf = int (q_RADIX_PIPPENGER_VARIANT>>1);
    let qhalf = Q_RADIX_PIPPENGER_VARIANT >> 1;
    // uint32_t mask = uint32_t (q_RADIX_PIPPENGER_VARIANT - 1);
    let mask = Q_RADIX_PIPPENGER_VARIANT - 1;

    // for (int i=0; i< h_BGMW95; ++i){
    for piece in ret_qhalf_expr.iter_mut() {
        // ret_qhalf_expr[i] = tmp.data[0] & mask;// we only need the bit-wise xor with the last 32-bit of tmp.
        *piece = (a[0] & (mask as u64)) as i32;
        // tmp = tmp >> EXPONENT_OF_q_BGMW95;
        a = uint256_bithsift(&a, EXPONENT_OF_Q_BGMW95 as u64);
    }
    // for (int i=0; i< h_BGMW95 - 1; ++i){
    for i in 0..(H_BGMW95 - 1) {
        // if(ret_qhalf_expr[i] > qhalf){
        if ret_qhalf_expr[i] > qhalf.try_into().unwrap() {
            // ret_qhalf_expr[i] -= q_RADIX_PIPPENGER_VARIANT;
            ret_qhalf_expr[i] -= Q_RADIX_PIPPENGER_VARIANT as i32;
            // ret_qhalf_expr[i+1] +=1;
            ret_qhalf_expr[i + 1] += 1;
            // // system parameter makes sure ret_qhalf_expr[h-1]<= q/2.
        }
    }
}

trait ThreadPoolExt {
    fn joined_execute<'any, F>(&self, job: F)
    where
        F: FnOnce() + Send + 'any;
}

use core::mem::transmute;
use std::sync::{Mutex, Once};
use threadpool::ThreadPool;

pub fn da_pool() -> ThreadPool {
    static INIT: Once = Once::new();
    static mut POOL: *const Mutex<ThreadPool> = 0 as *const Mutex<ThreadPool>;

    INIT.call_once(|| {
        let pool = Mutex::new(ThreadPool::default());
        unsafe { POOL = transmute(Box::new(pool)) };
    });
    unsafe { (*POOL).lock().unwrap().clone() }
}

type Thunk<'any> = Box<dyn FnOnce() + Send + 'any>;

impl ThreadPoolExt for ThreadPool {
    fn joined_execute<'scope, F>(&self, job: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        // Bypass 'lifetime limitations by brute force. It works,
        // because we explicitly join the threads...
        self.execute(unsafe { transmute::<Thunk<'scope>, Thunk<'static>>(Box::new(job)) })
    }
}

fn bgmw(ret: &mut FsG1, npoints: usize, scalars: &[FsFr], table: &[blst_p1_affine]) {
    // std::array< int, h_BGMW95> ret_qhalf_expr;
    let mut ret_qhalf_expr = [0i32; H_BGMW95];

    // uint64_t npoints = N_POINTS*h_BGMW95;

    // int* scalars;
    // scalars = new int [npoints];
    let mut scalars_out = vec![0i32; npoints * H_BGMW95];

    // unsigned char* booth_signs; // it acts as a bool type
    // booth_signs = new unsigned char [npoints];
    let mut booth_signs = vec![0u8; npoints * H_BGMW95];

    // blst_p1_affine** points_ptr;
    // points_ptr = new blst_p1_affine* [npoints];
    // let mut points_ptr = vec![ptr::null(); npoints * H_BGMW95];

    // FIXME: this formula only works when npoints is power of two
    let n_exp = npoints.leading_zeros();

    // idk, looks like BLS_MODULUS, but not sure
    const R_GROUP_ORDER: [u64; 4] = [
        0xffffffff00000001,
        0x53bda402fffe5bfe,
        0x3339d80809a1d805,
        0x73eda753299d7d48,
    ];
    // // This is only for BLS12-381 curve
    // if (N_EXP == 13 || N_EXP == 14 || N_EXP == 16 || N_EXP == 17){
    if n_exp == 13 || n_exp == 14 || n_exp == 16 || n_exp == 17 {
        // uint64_t  tt = uint64_t(1) << 62;
        let tt: u64 = 1u64 << 62;

        // for(int i = 0; i< N_POINTS; ++i){
        for (i, &aa) in scalars.iter().enumerate().take(npoints) {
            // uint256_t aa = scalars_array[i];

            let mut aa_scalar = {
                let mut scalar = blst_scalar::default();
                let mut arr = [0u64; 4];
                unsafe {
                    blst_scalar_from_fr(&mut scalar, &aa.0);
                    blst_uint64_from_scalar(arr.as_mut_ptr(), &scalar);
                };

                arr
            };

            // bool condition =  (aa.data[3] > tt); // a > 0.5*q*q**(h-1)
            let condition = aa_scalar[3] > tt;
            // if (condition == true) {
            if condition {
                // aa = r_GROUP_ORDER - aa;
                aa_scalar = uint256_sub(&R_GROUP_ORDER, &aa_scalar);
            }

            trans_uint256_to_qhalf_expr(&mut ret_qhalf_expr, aa_scalar);

            // if (condition == true) {
            if condition {
                // for(int j = 0; j< h_BGMW95; ++j){
                for (j, piece) in ret_qhalf_expr.iter().enumerate() {
                    // size_t idx = i*h_BGMW95 + j;
                    let idx = i * H_BGMW95 + j;
                    // scalars[idx]  = ret_qhalf_expr[j];
                    scalars_out[idx] = *piece;
                    // points_ptr[idx] =  PRECOMPUTATION_POINTS_LIST_BGMW95 + idx;
                    // points_ptr[idx] = table.as_ptr().wrapping_add(idx);

                    // if ( scalars[idx] > 0) {
                    if scalars_out[idx] > 0 {
                        // booth_signs[idx] = 1;
                        booth_signs[idx] = 1;
                    } else {
                        // scalars[idx] = - scalars[idx];
                        scalars_out[idx] = -scalars_out[idx];
                        // booth_signs[idx] = 0;
                        booth_signs[idx] = 0;
                    }
                }
            } else {
                // for(int j = 0; j< h_BGMW95; ++j){
                for (j, piece) in ret_qhalf_expr.iter().enumerate() {
                    // size_t idx = i*h_BGMW95 + j;
                    let idx = i * H_BGMW95 + j;
                    // scalars[idx]  = ret_qhalf_expr[j];
                    scalars_out[idx] = *piece;
                    // points_ptr[idx] =  PRECOMPUTATION_POINTS_LIST_BGMW95 + idx;
                    // points_ptr[idx] = table.as_ptr().wrapping_add(idx);

                    // if ( scalars[idx] > 0) {
                    if scalars_out[idx] > 0 {
                        // booth_signs[idx] = 0;
                        booth_signs[idx] = 0;
                    } else {
                        // scalars[idx] = - scalars[idx];
                        scalars_out[idx] = -scalars_out[idx];
                        // booth_signs[idx] = 1;
                        booth_signs[idx] = 1;
                    }
                }
            }
        }
    } else {
        // for(int i = 0; i< N_POINTS; ++i){
        for (i, fr) in scalars.iter().enumerate().take(npoints) {
            let scalar = {
                let mut scalar = blst_scalar::default();
                let mut arr = [0u64; 4];
                unsafe {
                    blst_scalar_from_fr(&mut scalar, &fr.0);
                    blst_uint64_from_scalar(arr.as_mut_ptr(), &scalar);
                };

                arr
            };

            // trans_uint256_t_to_qhalf_expr(ret_qhalf_expr, scalars_array[i]);
            trans_uint256_to_qhalf_expr(&mut ret_qhalf_expr, scalar);

            // for(int j = 0; j< h_BGMW95; ++j){
            for (j, piece) in ret_qhalf_expr.iter().enumerate() {
                // size_t idx = i*h_BGMW95 + j;
                let idx = i * H_BGMW95 + j;
                // scalars[idx]  = ret_qhalf_expr[j];
                scalars_out[idx] = *piece;
                // points_ptr[idx] =  PRECOMPUTATION_POINTS_LIST_BGMW95 + idx;
                // points_ptr[idx] = table.as_ptr().wrapping_add(idx);
                // if ( scalars[idx] > 0) {
                if scalars_out[idx] > 0 {
                    // booth_signs[idx] = 0;
                    booth_signs[idx] = 0;
                } else {
                    // scalars[idx] = -scalars[idx];
                    scalars_out[idx] = -scalars_out[idx];
                    // booth_signs[idx] = 1;
                    booth_signs[idx] = 1;
                }
            }
        }
    }

    // blst_p1 ret; // Mont coordinates

    // blst_p1xyzz* buckets;
    // int qhalf = int(q_RADIX_PIPPENGER_VARIANT>>1);
    let qhalf = Q_RADIX_PIPPENGER_VARIANT >> 1;
    // buckets = new blst_p1xyzz [qhalf + 1];
    // blst_p1_tile_pippenger_BGMW95(&ret, \
    //                                 points_ptr, \
    //                                 npoints, \
    //                                 scalars, booth_signs,\
    //                                 buckets,\
    //                                 EXPONENT_OF_q_BGMW95);
    // (1usize << (q_exponent - 1)) + 1

    // blst_p1s_tile_pippenger(
    //     grid[work].1.as_ptr(),
    //     &p[0],
    //     grid[work].0.dx,
    //     &s[0],
    //     nbits,
    //     &mut scratch[0],
    //     y,
    //     window,
    // );

    #[cfg(feature = "parallel")]
    {
        let pool = da_pool();
        let ncpus = pool.max_count();
        let n_workers = core::cmp::min(ncpus, H_BGMW95);

        let (tx, rx) = mpsc::channel();
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..n_workers {
            let tx = tx.clone();
            let counter = counter.clone();

            let scalars = &scalars_out;
            let booth_signs = &booth_signs;

            pool.joined_execute(move || {
                let mut buckets = vec![
                    P1XYZZ {
                        x: blst_fp { l: [0u64; 6] },
                        y: blst_fp { l: [0u64; 6] },
                        zz: blst_fp { l: [0u64; 6] },
                        zzz: blst_fp { l: [0u64; 6] },
                    };
                    qhalf + 1
                ];

                loop {
                    let work = counter.fetch_add(1, Ordering::Relaxed);

                    buckets.iter_mut().for_each(|b| {
                        *b = P1XYZZ {
                            x: blst_fp { l: [0u64; 6] },
                            y: blst_fp { l: [0u64; 6] },
                            zz: blst_fp { l: [0u64; 6] },
                            zzz: blst_fp { l: [0u64; 6] },
                        }
                    });

                    if work >= H_BGMW95 {
                        break;
                    }

                    let begin = work * npoints;
                    let end = (work + 1) * npoints;
                    let mut point = FsG1::default();

                    bgmw_pippenger_tile(
                        &mut point,
                        &table[begin..end],
                        npoints,
                        &scalars[begin..end],
                        &booth_signs[begin..end],
                        &mut buckets,
                        EXPONENT_OF_Q_BGMW95,
                    );

                    // TODO: error handling here
                    tx.send(point).unwrap();
                }
            });
        }

        *ret = FsG1::default();
        for _ in 0..H_BGMW95 {
            let value = rx.recv().unwrap();

            *ret = ret.add(&value);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut buckets = vec![
            P1XYZZ {
                x: blst_fp { l: [0u64; 6] },
                y: blst_fp { l: [0u64; 6] },
                zz: blst_fp { l: [0u64; 6] },
                zzz: blst_fp { l: [0u64; 6] },
            };
            qhalf + 1
        ];

        bgmw_pippenger_tile(
            ret,
            table,
            npoints * H_BGMW95,
            &scalars_out,
            &booth_signs,
            &mut buckets,
            EXPONENT_OF_Q_BGMW95,
        );
    }
}

#[allow(unused)]
pub unsafe fn msm(
    ret: &mut FsG1,
    points: &[FsG1],
    npoints: usize,
    scalars: &[FsFr],
    nbits: usize,
    scratch: *mut limb_t,
    table: Option<&BGMWPreComputationList>,
) {
    if npoints == 1 {
        *ret = points[0].mul(&scalars[0]);

        return;
    }

    if npoints * size_of::<blst_p1_affine>() * 8 * 3 <= (144 * 1024) {
        let mut table = vec![blst_p1_affine::default(); npoints * 8];

        {
            let mut points_affine = vec![blst_p1_affine::default(); npoints];
            let points_arg: [*const blst_p1; 2] = [&points[0].0, ptr::null()];
            unsafe { blst_p1s_to_affine(points_affine.as_mut_ptr(), points_arg.as_ptr(), npoints) };

            let points_affine_arg: [*const blst_p1_affine; 2] =
                [points_affine.as_ptr(), ptr::null()];

            unsafe {
                blst_p1s_mult_wbits_precompute(
                    table.as_mut_ptr(),
                    4,
                    points_affine_arg.as_ptr(),
                    npoints,
                );
            }
        };

        {
            let mut blst_scalars = vec![blst_scalar::default(); npoints];

            for i in 0..npoints {
                unsafe { blst_scalar_from_fr(&mut blst_scalars[i], &scalars[i].0) };
            }

            let scalars_arg: [*const blst_scalar; 2] = [blst_scalars.as_ptr(), ptr::null()];

            unsafe {
                blst_p1s_mult_wbits(
                    &mut ret.0,
                    table.as_ptr(),
                    4,
                    npoints,
                    scalars_arg.as_ptr() as *const *const u8,
                    nbits,
                    null_mut(),
                );
            }
        }

        return;
    }

    if let Some(table) = table {
        unsafe { bgmw(ret, npoints, scalars, table.0.as_slice()) }
    } else {
        #[cfg(feature = "parallel")]
        {
            let affines = P1Affines::from(&points[0..npoints]);

            let mut scalar_bytes: Vec<u8> = Vec::with_capacity(npoints * 32);
            for bytes in scalars.iter().map(|b| {
                let mut scalar = blst_scalar::default();

                unsafe { blst_scalar_from_fr(&mut scalar, &b.0) }

                scalar.b
            }) {
                scalar_bytes.extend_from_slice(&bytes);
            }

            *ret = FsG1(affines.mult(&scalar_bytes, nbits));
        }

        #[cfg(not(feature = "parallel"))]
        unsafe {
            let mut p_affine = vec![blst_p1_affine::default(); npoints];
            let p_arg: [*const blst_p1; 2] = [&points[0].0, ptr::null()];
            unsafe {
                blst_p1s_to_affine(p_affine.as_mut_ptr(), p_arg.as_ptr(), npoints);
            }

            let mut scalar_bytes: Vec<u8> = Vec::with_capacity(npoints * 32);
            for bytes in scalars.iter().map(|b| {
                let mut scalar = blst_scalar::default();

                unsafe { blst_scalar_from_fr(&mut scalar, &b.0) }

                scalar.b
            }) {
                scalar_bytes.extend_from_slice(&bytes);
            }

            pippenger(
                &mut ret.0,
                &p_affine,
                npoints,
                &scalar_bytes,
                nbits,
                scratch,
                0,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::msm::booth_encode;

    #[test]
    fn booth_encode_must_produce_correct_results() {
        assert_eq!(booth_encode(0, 1), 0);
        assert_eq!(booth_encode(0, 5), 0);
        assert_eq!(booth_encode(1, 1), 1);
        assert_eq!(booth_encode(55, 5), 18446744073709551588);
    }
}
