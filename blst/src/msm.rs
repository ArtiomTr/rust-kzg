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

unsafe fn get_wval_limb(mut d: *const byte, off: usize, bits: usize) -> limb_t {
    let mut top = (off + bits - 1) / 8;
    let mut mask = (0 as limb_t).wrapping_sub(1);

    d = d.wrapping_add(off / 8);

    top = top.wrapping_sub((off / 8).wrapping_sub(1));

    let mut ret: limb_t = 0;
    let mut i: usize = 0;
    loop {
        ret |= (*d as limb_t & mask) << (8 * i);
        mask =
            (0 as limb_t).wrapping_sub(((i + 1).wrapping_sub(top) >> (usize::BITS - 1)) as limb_t);
        i += 1;
        d = d.wrapping_add((1 & mask).try_into().unwrap());
        if i >= 4 {
            break ret >> (off % 8);
        }
    }
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

unsafe fn p1s_bucket(
    buckets: *mut P1XYZZ,
    mut booth_idx: limb_t,
    wbits: usize,
    p: *const blst_p1_affine,
) {
    let booth_sign = (booth_idx >> wbits) & 1;
    booth_idx &= (1 << wbits) - 1;
    if booth_idx != 0 {
        booth_idx -= 1;
        p1_dadd_affine(
            buckets.wrapping_add(booth_idx.try_into().unwrap()),
            buckets.wrapping_add(booth_idx.try_into().unwrap()),
            p,
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

unsafe fn p1_integrate_buckets(out: *mut blst_p1, buckets: *mut P1XYZZ, wbits: usize) {
    // POINTonE1xyzz ret[1], acc[1];
    let mut ret = [P1XYZZ::default()];
    let mut acc = [P1XYZZ::default()];

    // size_t n = (size_t)1 << wbits;
    let mut n = (1usize << wbits) - 1;
    // vec_copy(acc, &buckets[--n], sizeof(acc));
    vec_copy(
        &mut acc as *mut P1XYZZ as *mut u8,
        buckets.wrapping_add(n) as *const u8,
        size_of::<P1XYZZ>(),
    );
    // vec_copy(ret, &buckets[n], sizeof(ret));
    vec_copy(
        &mut ret as *mut P1XYZZ as *mut u8,
        buckets.wrapping_add(n) as *const u8,
        size_of::<P1XYZZ>(),
    );
    // vec_zero(&buckets[n], sizeof(buckets[n]));
    vec_zero(buckets.wrapping_add(n) as *mut limb_t, size_of::<P1XYZZ>());

    // while (n--)
    loop {
        if n == 0 {
            break;
        }
        n -= 1;

        // POINTonE1xyzz_dadd(acc, acc, &buckets[n]);
        p1_dadd(acc.as_mut_ptr(), acc.as_ptr(), buckets.wrapping_add(n));
        // POINTonE1xyzz_dadd(ret, ret, acc);
        p1_dadd(ret.as_mut_ptr(), ret.as_ptr(), acc.as_ptr());
        // vec_zero(&buckets[n], sizeof(buckets[n]));
        vec_zero(buckets.wrapping_add(n) as *mut limb_t, size_of::<P1XYZZ>());
    }

    // POINTonE1xyzz_to_Jacobian(out, ret);
    // blst::blst_p1_from_jacobian(out, ret.as_ptr() as *const blst_p1);
    p1_to_jacobian(out, ret.as_ptr());
}

// unsafe fn p1_prefetch() {
// booth_idx &= (1 << wbits) - 1;
// if (booth_idx--)
//     vec_prefetch(&buckets[booth_idx], sizeof(buckets[booth_idx]));
// }

#[allow(clippy::too_many_arguments)]
unsafe fn p1s_tile_pippenger(
    ret: *mut blst_p1,
    mut points: *const *const blst_p1_affine,
    mut npoints: usize,
    mut scalars: *const *const byte,
    nbits: usize,
    buckets: *mut limb_t,
    mut bit0: usize,
    wbits: usize,
    cbits: usize,
) {
    // limb_t wmask, wval, wnxt;
    // size_t i, z, nbytes;
    // const byte *scalar = *scalars++;
    let scalar = *scalars;
    scalars = scalars.wrapping_add(1);
    // const POINTonE1_affine *point = *points++;
    let mut point = *points;
    points = points.wrapping_add(1);
    // nbytes = (nbits + 7) / 8;
    let nbytes = (nbits + 7) / 8;
    // wmask = ((limb_t)1 << (wbits + 1)) - 1;
    let wmask = ((1 as limb_t) << (wbits + 1)) - 1;
    // z = is_zero(bit0);
    let z = is_zero((bit0).try_into().unwrap());
    // bit0 -= z ^ 1;
    bit0 -= (z ^ 1) as usize;
    // wbits += z ^ 1;
    let wbits = wbits + (z ^ 1) as usize;
    // wval = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let wval = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    // wval = booth_encode(wval, cbits);
    let mut wval = booth_encode(wval, cbits);
    // scalar = *scalars ? *scalars++ : scalar + nbytes;
    let mut scalar = if (*scalars).is_null() {
        scalar.wrapping_add(nbytes)
    } else {
        let v = *scalars;
        scalars = scalars.wrapping_add(1);
        v
    };
    // wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    // wnxt = booth_encode(wnxt, cbits);
    let mut wnxt = booth_encode(wnxt, cbits);
    // npoints--;
    npoints -= 1;

    // POINTonE1_bucket(buckets, wval, cbits, point);
    p1s_bucket(buckets as *mut P1XYZZ, wval, cbits, point);
    // for (i = 1; i < npoints; i++)
    for _i in 1..npoints {
        // wval = wnxt;
        wval = wnxt;
        // scalar = *scalars ? *scalars++ : scalar + nbytes;
        scalar = if (*scalars).is_null() {
            scalar.wrapping_add(nbytes)
        } else {
            let v = *scalars;
            scalars = scalars.wrapping_add(1);
            v
        };
        // wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
        wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
        // wnxt = booth_encode(wnxt, cbits);
        wnxt = booth_encode(wnxt, cbits);
        // POINTonE1_prefetch(buckets, wnxt, cbits);
        // p1_prefetch(buckets, wnxt, cbits);
        // TODO:

        // point = *points ? *points++ : point + 1;
        point = if (*points).is_null() {
            point.wrapping_add(1)
        } else {
            let v = *points;
            points = points.wrapping_add(1);
            v
        };
        // POINTonE1_bucket(buckets, wval, cbits, point);
        p1s_bucket(buckets as *mut P1XYZZ, wval, cbits, point);
    }
    // point = *points ? *points++ : point + 1;
    point = if (*points).is_null() {
        point.wrapping_add(1)
    } else {
        *points
    };
    // POINTonE1_bucket(buckets, wnxt, cbits, point);
    p1s_bucket(buckets as *mut P1XYZZ, wnxt, cbits, point);
    // POINTonE1_integrate_buckets(ret, buckets, cbits - 1);
    p1_integrate_buckets(ret, buckets as *mut P1XYZZ, cbits - 1);
}

unsafe fn pippenger(
    ret: *mut blst_p1,
    points: *const *const blst_p1_affine,
    npoints: usize,
    scalars: *const *const byte,
    nbits: usize,
    buckets: *mut limb_t,
    window: usize,
) {
    // size_t i, wbits, cbits, bit0 = nbits;
    // POINTonE1 tile[1];
    let window = if window != 0 {
        window
    } else {
        pippenger_window_size(npoints)
    };

    vec_zero(buckets, size_of::<limb_t>() << (window - 1));
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
        blst_p1_add(ret, ret, tile.as_mut_ptr());
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

unsafe fn p1s_bucket_CHES(
    buckets: *mut P1XYZZ,
    booth_idx: limb_t,
    point: *const blst_p1_affine,
    booth_sign: u8,
) {
    p1_dadd_affine(
        buckets.wrapping_add(booth_idx.try_into().unwrap()),
        buckets.wrapping_add(booth_idx.try_into().unwrap()),
        point,
        booth_sign.into(),
    );
}

unsafe fn bgmw_pippenger_tile(
    ret: *mut blst_p1,
    mut points: *const *const blst_p1_affine,
    mut npoints: usize,
    mut scalars: *const i32,
    mut booth_signs: *const u8,
    buckets: *mut P1XYZZ,
    q_exponent: usize,
) {
    // POINTonE1 *ret, const POINTonE1_affine *const points[], size_t npoints, const int scalars[], const unsigned char booth_signs[], POINTonE1xyzz buckets[], size_t q_exponent

    // size_t bucket_set_size = (size_t)(1 << (q_exponent - 1)) + 1;
    let bucket_set_size = (1usize << (q_exponent - 1)) + 1;
    // vec_zero(buckets, sizeof(buckets[0]) * bucket_set_size);
    vec_zero(
        buckets as *mut limb_t,
        size_of::<P1XYZZ>() * bucket_set_size,
    );
    // vec_zero(ret, sizeof(*ret));
    vec_zero(ret as *mut limb_t, size_of::<blst_p1>());
    // int booth_idx, booth_idx_nxt;
    // size_t i;
    // unsigned char booth_sign;
    // const POINTonE1_affine *point = *points++;
    let mut point = *points;
    points = points.wrapping_add(1);
    // booth_idx = *scalars++;
    let mut booth_idx = *scalars;
    scalars = scalars.wrapping_add(1);
    // booth_sign = *booth_signs++;
    let mut booth_sign = *booth_signs;
    booth_signs = booth_signs.wrapping_add(1);
    // booth_idx_nxt = *scalars++;
    let mut booth_idx_nxt = *scalars;
    scalars = scalars.wrapping_add(1);

    // if (booth_idx)
    if booth_idx != 0 {
        // POINTonE1_bucket_CHES(buckets, booth_idx, point, booth_sign);
        p1s_bucket_CHES(buckets, booth_idx as limb_t, point, booth_sign);
    }
    // --npoints;
    npoints -= 1;
    // for (i = 1; i < npoints; ++i)
    for _i in 1..npoints {
        // booth_idx = booth_idx_nxt;
        booth_idx = booth_idx_nxt;
        // booth_idx_nxt = *scalars++;
        booth_idx_nxt = *scalars;
        scalars = scalars.wrapping_add(1);

        // TODO:
        // POINTonE1_prefetch_CHES(buckets, booth_idx_nxt);

        // point = *points++;
        point = *points;
        points = points.wrapping_add(1);
        // booth_sign = *booth_signs++;
        booth_sign = *booth_signs;
        booth_signs = booth_signs.wrapping_add(1);
        // if (booth_idx)
        if booth_idx != 0 {
            // POINTonE1_bucket_CHES(buckets, booth_idx, point, booth_sign);
            p1s_bucket_CHES(buckets, booth_idx as limb_t, point, booth_sign);
        }
    }
    // point = *points;
    point = *points;
    // booth_sign = *booth_signs;
    booth_sign = *booth_signs;
    // POINTonE1_bucket_CHES(buckets, booth_idx_nxt, point, booth_sign);
    p1s_bucket_CHES(buckets, booth_idx_nxt as limb_t, point, booth_sign);

    // ++buckets;
    let buckets = buckets.wrapping_add(1);

    // POINTonE1_integrate_buckets(ret, buckets, q_exponent - 1);
    p1_integrate_buckets(ret, buckets, q_exponent - 1);
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
    for i in 0..H_BGMW95 {
        // ret_qhalf_expr[i] = tmp.data[0] & mask;// we only need the bit-wise xor with the last 32-bit of tmp.
        ret_qhalf_expr[i] = (a[0] & (mask as u64)) as i32;
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

unsafe fn bgmw(ret: &mut FsG1, npoints: usize, scalars: &[FsFr], table: &[blst_p1_affine]) {
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
    let mut points_ptr = vec![ptr::null(); npoints * H_BGMW95];

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
        for i in 0..npoints {
            // uint256_t aa = scalars_array[i];
            let aa = scalars[i];

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
                for j in 0..H_BGMW95 {
                    // size_t idx = i*h_BGMW95 + j;
                    let idx = i * H_BGMW95 + j;
                    // scalars[idx]  = ret_qhalf_expr[j];
                    scalars_out[idx] = ret_qhalf_expr[j];
                    // points_ptr[idx] =  PRECOMPUTATION_POINTS_LIST_BGMW95 + idx;
                    points_ptr[idx] = table.as_ptr().wrapping_add(idx);

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
                for j in 0..H_BGMW95 {
                    // size_t idx = i*h_BGMW95 + j;
                    let idx = i * H_BGMW95 + j;
                    // scalars[idx]  = ret_qhalf_expr[j];
                    scalars_out[idx] = ret_qhalf_expr[j];
                    // points_ptr[idx] =  PRECOMPUTATION_POINTS_LIST_BGMW95 + idx;
                    points_ptr[idx] = table.as_ptr().wrapping_add(idx);

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
        for i in 0..npoints {
            let scalar = {
                let mut scalar = blst_scalar::default();
                let mut arr = [0u64; 4];
                unsafe {
                    blst_scalar_from_fr(&mut scalar, &scalars[i].0);
                    blst_uint64_from_scalar(arr.as_mut_ptr(), &scalar);
                };

                arr
            };

            // trans_uint256_t_to_qhalf_expr(ret_qhalf_expr, scalars_array[i]);
            trans_uint256_to_qhalf_expr(&mut ret_qhalf_expr, scalar);

            // for(int j = 0; j< h_BGMW95; ++j){
            for j in 0..H_BGMW95 {
                // size_t idx = i*h_BGMW95 + j;
                let idx = i * H_BGMW95 + j;
                // scalars[idx]  = ret_qhalf_expr[j];
                scalars_out[idx] = ret_qhalf_expr[j];
                // points_ptr[idx] =  PRECOMPUTATION_POINTS_LIST_BGMW95 + idx;
                points_ptr[idx] = table.as_ptr().wrapping_add(idx);
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
    let mut buckets = vec![0u8; (qhalf + 1) * size_of::<P1XYZZ>()];

    // blst_p1_tile_pippenger_BGMW95(&ret, \
    //                                 points_ptr, \
    //                                 npoints, \
    //                                 scalars, booth_signs,\
    //                                 buckets,\
    //                                 EXPONENT_OF_q_BGMW95);
    bgmw_pippenger_tile(
        &mut ret.0,
        points_ptr.as_ptr(),
        npoints * H_BGMW95,
        scalars_out.as_ptr(),
        booth_signs.as_ptr(),
        buckets.as_mut_ptr() as *mut P1XYZZ,
        EXPONENT_OF_Q_BGMW95,
    );
    // delete[] buckets;
    // delete[] points_ptr;
    // delete[] booth_signs;
    // delete[] scalars;

    // blst_p1_affine res_affine;
    // blst_p1_to_affine( &res_affine, &ret);
    // return res_affine;
}

#[allow(unused)]
pub fn msm(
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
        let mut p_affine = vec![blst_p1_affine::default(); npoints];
        let mut p_scalars = vec![blst_scalar::default(); npoints];

        let p_arg: [*const blst_p1; 2] = [&points[0].0, ptr::null()];
        unsafe {
            blst_p1s_to_affine(p_affine.as_mut_ptr(), p_arg.as_ptr(), npoints);
        }

        for i in 0..npoints {
            unsafe { blst_scalar_from_fr(&mut p_scalars[i], &scalars[i].0) };
        }

        let scalars_arg: [*const blst_scalar; 2] = [p_scalars.as_ptr(), ptr::null()];
        let points_arg: [*const blst_p1_affine; 2] = [p_affine.as_ptr(), ptr::null()];

        unsafe {
            pippenger(
                &mut ret.0,
                points_arg.as_ptr(),
                npoints,
                scalars_arg.as_ptr() as *const *const u8,
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
