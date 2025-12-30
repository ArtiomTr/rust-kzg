//! MSM Utilities
//!
//! This module contains shared utilities used by various MSM implementations:
//! - Point representation (P1XYZZ)
//! - Booth encoding for signed digit representation
//! - Window size heuristics
//! - Batch inversion and point operations
//! - Parallel breakdown calculations

use core::mem::size_of;
use core::ops::Neg;

use alloc::vec;
use alloc::vec::Vec;

use crate::{G1, G1Affine, G1Fp, G1GetFp, G1ProjAddAffine, Scalar256};

// ============================================================================
// POINT REPRESENTATION
// ============================================================================

/// Extended Jacobian coordinates (x, y, zz, zzz)
///
/// This representation is used for efficient bucket accumulation in Pippenger.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct P1XYZZ<TFp: G1Fp> {
    pub x: TFp,
    pub y: TFp,
    pub zzz: TFp,
    pub zz: TFp,
}

// ============================================================================
// MEMORY UTILITIES
// ============================================================================

#[inline(always)]
pub fn type_zero<T>(ret: &mut T) {
    let rp = ret as *mut T as *mut u64;
    let num = size_of::<T>() / size_of::<u64>();

    for i in 0..num {
        unsafe {
            *rp.wrapping_add(i) = 0;
        }
    }
}

pub const fn is_zero(val: u64) -> u64 {
    (!val & (val.wrapping_sub(1))) >> (u64::BITS - 1)
}

#[inline(always)]
pub fn vec_zero_rt(ret: *mut u64, mut num: usize) {
    num /= size_of::<usize>();
    for i in 0..num {
        unsafe {
            *ret.add(i) = 0;
        }
    }
}

#[inline(always)]
pub fn vec_is_zero(a: *const u8, num: usize) -> u64 {
    let ap = a as *const u64;
    let num = num / size_of::<u64>();

    let mut acc: u64 = 0;
    for i in 0..num {
        unsafe {
            acc |= *ap.wrapping_add(i);
        }
    }

    is_zero(acc)
}

#[inline(always)]
pub fn type_is_zero<T>(a: &T) -> u64 {
    let ap = a as *const T as *const u64;
    let num = size_of::<T>() / size_of::<u64>();

    let mut acc: u64 = 0;
    for i in 0..num {
        unsafe {
            acc |= *ap.wrapping_add(i);
        }
    }

    is_zero(acc)
}

#[inline(always)]
fn vec_copy(ret: *mut u8, a: *const u8, num: usize) {
    let rp = ret as *mut u64;
    let ap = a as *const u64;

    let num = num / size_of::<u64>();

    for i in 0..num {
        unsafe {
            *rp.wrapping_add(i) = *ap.wrapping_add(i);
        }
    }
}

// ============================================================================
// POINT ARITHMETIC (Extended Jacobian)
// ============================================================================

/// Convert P1XYZZ to standard Jacobian coordinates
pub fn p1_to_jacobian<TG1: G1 + G1GetFp<TFp>, TFp: G1Fp>(out: &mut TG1, input: &P1XYZZ<TFp>) {
    *out.x_mut() = input.x.mul_fp(&input.zz);
    *out.y_mut() = input.y.mul_fp(&input.zzz);
    *out.z_mut() = input.zz;
}

/// Add affine point to P1XYZZ accumulator
pub fn p1_dadd_affine<TG1: G1, TFp: G1Fp, TG1Affine: G1Affine<TG1, TFp>>(
    out: &mut P1XYZZ<TFp>,
    p2: &TG1Affine,
    subtract: bool,
) {
    if p2.is_zero() {
        return;
    } else if vec_is_zero(&out.zzz as *const TFp as *const u8, 2 * size_of::<TFp>()) != 0 {
        vec_copy(
            &mut (out.x) as *mut TFp as *mut u8,
            ((*p2).x()) as *const TFp as *const u8,
            2 * size_of::<TFp>(),
        );

        out.zzz = TFp::bls12_381_rx_p();
        if subtract {
            out.zzz.neg_assign();
        }

        out.zz = TFp::bls12_381_rx_p();
        return;
    }

    let mut p = p2.x().mul_fp(&out.zz);
    let mut r = p2.y().mul_fp(&out.zzz);
    if subtract {
        r.neg_assign();
    }
    p.sub_assign_fp(&out.x);
    r.sub_assign_fp(&out.y);
    if type_is_zero(&p) == 0 {
        let pp = p.square();
        let ppp = pp.mul_fp(&p);
        let mut q = out.x.mul_fp(&pp);
        out.x = r.square();
        p = q.add_fp(&q);
        out.x.sub_assign_fp(&ppp);
        out.x.sub_assign_fp(&p);
        q.sub_assign_fp(&out.x);
        q.mul_assign_fp(&r);
        out.y.mul_assign_fp(&ppp);
        out.y = q.sub_fp(&out.y);
        out.zz.mul_assign_fp(&pp);
        out.zzz.mul_assign_fp(&ppp);
    } else if type_is_zero(&r) != 0 {
        let mut u = p2.y().add_fp(p2.y());
        out.zz = u.square();
        out.zzz = out.zz.mul_fp(&u);
        let mut s = p2.x().mul_fp(&out.zz);
        let mut m = p2.x().square();
        m = m.add_fp(&m).add_fp(&m);
        out.x = m.square();
        u = s.add_fp(&s);
        out.x.sub_assign_fp(&u);
        out.y = out.zzz.mul_fp(p2.y());
        s.sub_assign_fp(&out.x);
        s.mul_assign_fp(&m);
        out.y = s.sub_fp(&out.y);
        if subtract {
            out.zzz.neg_assign();
        }
    } else {
        vec_zero_rt(
            &mut out.zzz as *mut TFp as *mut u64,
            2 * core::mem::size_of_val(&out.zzz),
        );
    }
}

/// Add two P1XYZZ points
pub fn p1_dadd<TFp: G1Fp>(out: &mut P1XYZZ<TFp>, p2: &P1XYZZ<TFp>) {
    if vec_is_zero(&p2.zzz as *const TFp as *const u8, 2 * size_of::<TFp>()) != 0 {
        return;
    } else if vec_is_zero(&out.zzz as *const TFp as *const u8, 2 * size_of::<TFp>()) != 0 {
        *out = *p2;
        return;
    }

    let mut u = out.x.mul_fp(&p2.zz);
    let mut s = out.y.mul_fp(&p2.zzz);
    let mut p = p2.x.mul_fp(&out.zz);
    let mut r = p2.y.mul_fp(&out.zzz);

    p.sub_assign_fp(&u);
    r.sub_assign_fp(&s);

    if type_is_zero(&p) == 0 {
        let pp = p.square();
        let ppp = pp.mul_fp(&p);
        let mut q = u.mul_fp(&pp);
        out.x = r.square();
        p = q.add_fp(&q);
        out.x.sub_assign_fp(&ppp);
        out.x.sub_assign_fp(&p);
        q.sub_assign_fp(&out.x);
        q.mul_assign_fp(&r);
        out.y = s.mul_fp(&ppp);
        out.y = q.sub_fp(&out.y);
        out.zz.mul_assign_fp(&p2.zz);
        out.zzz.mul_assign_fp(&p2.zzz);
        out.zz.mul_assign_fp(&pp);
        out.zzz.mul_assign_fp(&ppp);
    } else if type_is_zero(&r) != 0 {
        u = out.y.add_fp(&out.y);
        let v = u.square();
        let w = v.mul_fp(&u);
        s = out.x.mul_fp(&v);
        let mut m = out.x.square();
        m = m.add_fp(&m).add_fp(&m);
        out.x = m.square();
        u = s.add_fp(&s);
        out.x.sub_assign_fp(&u);
        out.y = w.mul_fp(&out.y);
        s.sub_assign_fp(&out.x);
        s.mul_assign_fp(&m);
        out.y = s.sub_fp(&out.y);
        out.zz.mul_assign_fp(&v);
        out.zzz.mul_assign_fp(&w);
    } else {
        vec_zero_rt(&mut out.zzz as *mut TFp as *mut u64, 2 * size_of::<TFp>());
    }
}

// ============================================================================
// SCALAR BIT EXTRACTION
// ============================================================================

/// Extract `bits` from the beginning of `d` array, with offset `off`.
///
/// This function is used to extract N bits from the scalar, decomposing it into q-ary representation.
/// This works because `q` is `2^bits`, so extracting `bits` from scalar will break it into the correct representation.
///
/// # Arguments
///
/// * `d`    - byte array, from which bits will be extracted
/// * `off`  - index of first bit, that will be extracted
/// * `bits` - number of bits to extract (up to 25)
pub fn get_wval_limb(d: &Scalar256, off: usize, bits: usize) -> u64 {
    let mut d = d.as_u8();
    let top = ((off + bits - 1) / 8).wrapping_sub((off / 8).wrapping_sub(1));
    d = &d[off / 8..];
    let mut mask = u64::MAX;
    let mut ret: u64 = 0;
    for i in 0..4usize {
        ret |= (d[0] as u64 & mask) << (8 * i);

        mask = 0u64.wrapping_sub(((i + 1).wrapping_sub(top) >> (usize::BITS - 1)) as u64);
        d = &d[(1 & mask).try_into().unwrap()..];
    }
    ret >> (off % 8)
}

// ============================================================================
// BOOTH ENCODING (Pippenger variant)
// ============================================================================

/// Window value encoding that utilizes the fact that -P is trivially
/// calculated, which allows to halve the size of the pre-computed table.
///
/// This is attributed to A. D. Booth.
pub const fn booth_encode(wval: u64, sz: usize) -> u64 {
    let mask = 0u64.wrapping_sub(wval >> sz);

    let wval = (wval + 1) >> 1;
    (wval ^ mask).wrapping_sub(mask)
}

/// Decode bucket index and move point to corresponding bucket
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
pub fn booth_decode<TG1: G1, TFp: G1Fp, TG1Affine: G1Affine<TG1, TFp>>(
    buckets: &mut [P1XYZZ<TFp>],
    mut booth_idx: u64,
    wbits: usize,
    p: &TG1Affine,
) {
    let booth_sign: bool = ((booth_idx >> wbits) & 1) != 0;
    booth_idx &= (1 << wbits) - 1;
    if booth_idx != 0 {
        p1_dadd_affine(&mut buckets[(booth_idx - 1) as usize], p, booth_sign);
    }
}

// ============================================================================
// BOOTH ENCODING (Wbits variant)
// ============================================================================

/// Get Booth index for wbits MSM algorithm.
///
/// Code was taken from: https://github.com/privacy-scaling-explorations/halo2curves
pub fn get_booth_index(window_index: usize, window_size: usize, el: &[u8]) -> i32 {
    // Booth encoding:
    // * step by `window` size
    // * slice by size of `window + 1`
    // * each window overlap by 1 bit
    // * append a zero bit to the least significant end
    // Indexing rule for example window size 3 where we slice by 4 bits:
    // `[0, +1, +1, +2, +2, +3, +3, +4, -4, -3, -3 -2, -2, -1, -1, 0]`
    // So we can reduce the bucket size without preprocessing scalars
    // and remembering them as in classic signed digit encoding

    let skip_bits = (window_index * window_size).saturating_sub(1);
    let skip_bytes = skip_bits / 8;

    // fill into a u32
    let mut v: [u8; 4] = [0; 4];
    for (dst, src) in v.iter_mut().zip(el.iter().skip(skip_bytes)) {
        *dst = *src
    }
    let mut tmp = u32::from_le_bytes(v);

    // pad with one 0 if slicing the least significant window
    if window_index == 0 {
        tmp <<= 1;
    }

    // remove further bits
    tmp >>= skip_bits - (skip_bytes * 8);
    // apply the booth window
    tmp &= (1 << (window_size + 1)) - 1;

    let sign = tmp & (1 << window_size) == 0;

    // div ceil by 2
    tmp = (tmp + 1) >> 1;

    // find the booth action index
    if sign {
        tmp as i32
    } else {
        ((!(tmp - 1) & ((1 << window_size) - 1)) as i32).neg()
    }
}

// ============================================================================
// WINDOW SIZE HEURISTICS
// ============================================================================

pub const fn num_bits(l: usize) -> usize {
    8 * core::mem::size_of::<usize>() - l.leading_zeros() as usize
}

/// Calculate optimal window size for Pippenger MSM.
///
/// This approximates the minimum of:
/// y = ceil(255/w) * (npoints + 2^w + w + 1)
///
/// Parts:
///   - ceil(255/w): number of parts in decomposed scalar
///   - npoints: bucket accumulation cost
///   - 2^w: bucket aggregation cost
///   - w + 1: final doubling and addition
pub fn pippenger_window_size(npoints: usize) -> usize {
    option_env!("WINDOW_SIZE")
        .map(|v| {
            v.parse()
                .expect("WINDOW_SIZE environment variable must be valid number")
        })
        .unwrap_or({
            let wbits = num_bits(npoints);

            if wbits > 13 {
                return wbits - 4;
            }
            if wbits > 5 {
                return wbits - 3;
            }
            2
        })
}

/// Get window size for wbits algorithm
pub fn get_wbits_window_size() -> usize {
    option_env!("WINDOW_SIZE")
        .map(|v| {
            v.parse()
                .expect("WINDOW_SIZE environment variable must be valid number")
        })
        .unwrap_or(8)
}

// ============================================================================
// PARALLEL BREAKDOWN
// ============================================================================

/// Calculate optimal grid breakdown for parallel Pippenger.
///
/// Returns (nx, ny, window) where:
/// - nx: number of divisions along the points axis
/// - ny: number of divisions along the scalar bits axis
/// - window: adjusted window size
#[cfg(feature = "parallel")]
pub fn parallel_breakdown(window: usize, ncpus: usize) -> (usize, usize, usize) {
    const NBITS: usize = 255;

    option_env!("WINDOW_NX")
        .map(|v| {
            v.parse()
                .expect("WINDOW_NX environment variable must be valid number")
        })
        .map(|nx| {
            let ny = NBITS / window + 1;
            (nx, ny, NBITS / ny + 1)
        })
        .unwrap_or({
            let mut nx: usize;
            let mut wnd: usize;

            if NBITS > window * ncpus {
                nx = 1;
                wnd = num_bits(ncpus / 4);
                if (window + wnd) > 18 {
                    wnd = window - wnd;
                } else {
                    wnd = (NBITS / window).div_ceil(ncpus);
                    if (NBITS / (window + 1)).div_ceil(ncpus) < wnd {
                        wnd = window + 1;
                    } else {
                        wnd = window;
                    }
                }
            } else {
                nx = 2;
                wnd = window - 2;
                while (NBITS / wnd + 1) * nx < ncpus {
                    nx += 1;
                    wnd = window - num_bits(3 * nx / 2);
                }
                nx -= 1;
                wnd = window - num_bits(3 * nx / 2);
            }
            let ny = NBITS / wnd + 1;
            wnd = NBITS / ny + 1;

            (nx, ny, wnd)
        })
}

/// Alias for parallel_breakdown for backward compatibility
#[cfg(feature = "parallel")]
pub fn breakdown(window: usize, ncpus: usize) -> (usize, usize, usize) {
    parallel_breakdown(window, ncpus)
}

// ============================================================================
// BATCH AFFINE OPERATIONS
// ============================================================================

/// This is the threshold to which batching the inversions in affine
/// formula costs more than doing mixed addition.
pub const BATCH_INVERSE_THRESHOLD: usize = 16;

/// Chooses between point addition and point doubling based on the input points.
///
/// Note: This does not handle the case where p1 == -p2.
#[inline(always)]
pub fn choose_add_or_double<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    p1: TG1Affine,
    p2: TG1Affine,
) -> TG1Fp {
    if p1 == p2 {
        p2.y().double()
    } else {
        p2.x().sub_fp(p1.x())
    }
}

/// Given a vector of field elements {v_i}, compute the vector {v_i^(-1)}
///
/// A scratchpad is used to avoid excessive allocations in the case that this method is
/// called repeatedly.
///
/// Uses Montgomery's Trick for efficient batch inversion.
pub fn batch_inverse_scratch_pad<F: G1Fp>(v: &mut [F], scratchpad: &mut Vec<F>) {
    // Clear the scratchpad and ensure it has enough capacity
    scratchpad.clear();
    scratchpad.reserve(v.len());

    // First pass: compute [a, ab, abc, ...]
    let mut tmp = F::one();
    for f in v.iter() {
        tmp = tmp.mul_fp(f);
        scratchpad.push(tmp);
    }

    // Invert `tmp`.
    tmp = tmp
        .inverse()
        .expect("guaranteed to be non-zero since we filtered out zero field elements");

    // Second pass: iterate backwards to compute inverses
    for (f, s) in v
        .iter_mut()
        // Backwards
        .rev()
        // Backwards, skip last element, fill in one for last term.
        .zip(scratchpad.iter().rev().skip(1).chain(Some(&F::one())))
    {
        // tmp := tmp * f; f := tmp * s = 1/f
        let new_tmp = tmp.mul_fp(f);
        *f = tmp.mul_fp(s);
        tmp = new_tmp;
    }
}

/// Adds two elliptic curve points using the point addition/doubling formula.
///
/// Note: The inversion is precomputed and passed as a parameter.
#[inline(always)]
pub fn point_add_double<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    p1: TG1Affine,
    p2: TG1Affine,
    inv: &TG1Fp,
) -> TG1Affine {
    let lambda = if p1 == p2 {
        p1.x().square().mul3().mul_fp(inv)
    } else {
        p2.y().sub_fp(p1.y()).mul_fp(inv)
    };

    let x = lambda.square().sub_fp(p1.x()).sub_fp(p2.x());
    let y = lambda.mul_fp(&p1.x().sub_fp(&x)).sub_fp(p1.y());

    TG1Affine::from_xy(x, y)
}

/// Performs multi-batch addition of multiple sets of elliptic curve points.
///
/// This function efficiently adds multiple sets of points amortizing the cost of the
/// inversion over all of the sets, using the same binary tree approach with striding
/// as the single-batch version.
pub fn multi_batch_addition_binary_tree_stride<
    TG1: G1,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
>(
    mut multi_points: Vec<Vec<TG1Affine>>,
) -> Vec<TG1> {
    multi_points
        .iter_mut()
        .for_each(|points| points.retain(|p| !p.is_infinity()));
    let total_num_points: usize = multi_points.iter().map(|p| p.len()).sum();
    let mut scratchpad = Vec::with_capacity(total_num_points);

    // Find the largest buckets, this will be the bottleneck for the number of iterations
    let mut max_bucket_length = 0;
    for points in multi_points.iter() {
        max_bucket_length = core::cmp::max(max_bucket_length, points.len());
    }

    // Compute the total number of "unit of work"
    #[inline(always)]
    fn compute_threshold<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
        points: &[Vec<TG1Affine>],
    ) -> usize {
        points
            .iter()
            .map(|p| {
                if p.len() % 2 == 0 {
                    p.len() / 2
                } else {
                    (p.len() - 1) / 2
                }
            })
            .sum()
    }

    let mut denominators = Vec::with_capacity(max_bucket_length);
    let mut total_amount_of_work = compute_threshold(&multi_points);

    let mut sums = vec![TG1::identity(); multi_points.len()];

    assert!(
        BATCH_INVERSE_THRESHOLD >= 2,
        "THRESHOLD cannot be below the number of points needed for group addition"
    );

    while total_amount_of_work > BATCH_INVERSE_THRESHOLD {
        // For each point, we check if they are odd and pop off one of the points
        for (points, sum) in multi_points.iter_mut().zip(sums.iter_mut()) {
            // Make the number of points even
            if points.len() % 2 != 0 {
                TG1ProjAddAffine::add_or_double_assign_affine(sum, &points.pop().unwrap());
            }
        }

        denominators.clear();

        // For each pair of points over all vectors, collect denominators
        for points in multi_points.iter_mut() {
            if points.len() < 2 {
                continue;
            }

            *points = points
                .chunks_exact(2)
                .filter(|v| v[0] != v[1].neg())
                .flat_map(|v| v)
                .cloned()
                .collect::<Vec<_>>();

            for i in (0..=points.len() - 2).step_by(2) {
                denominators.push(choose_add_or_double(points[i], points[i + 1]));
            }
        }

        batch_inverse_scratch_pad(&mut denominators, &mut scratchpad);

        let mut denominators_offset = 0;

        for points in multi_points.iter_mut() {
            if points.len() < 2 {
                continue;
            }

            for (i, inv) in (0..=points.len() - 2)
                .step_by(2)
                .zip(&denominators[denominators_offset..])
            {
                let p1 = points[i];
                let p2 = points[i + 1];
                points[i / 2] = point_add_double(p1, p2, inv);
            }

            let num_points = points.len() / 2;
            // The latter half of the vector is now unused
            points.truncate(num_points);
            denominators_offset += num_points
        }

        total_amount_of_work = compute_threshold(&multi_points);
    }

    for (sum, points) in sums.iter_mut().zip(multi_points) {
        for point in points {
            TG1ProjAddAffine::add_or_double_assign_affine(sum, &point);
        }
    }

    sums
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn booth_encode_must_produce_correct_results() {
        assert_eq!(booth_encode(0, 1), 0);
        assert_eq!(booth_encode(0, 5), 0);
        assert_eq!(booth_encode(1, 1), 1);
        assert_eq!(booth_encode(55, 5), 18446744073709551588);
    }

    #[test]
    fn get_wval_limb_example_1() {
        let val = get_wval_limb(
            &Scalar256 {
                data: [0b01010111u64, 0u64, 0u64, 0u64],
            },
            0,
            4,
        );
        assert_eq!(val, 0b01010111);
        // if you want to get value containing only extracted bits and zeros, do bitwise and on return value with mask:
        assert_eq!(val & 0b00001111, 0b00000111);
    }

    #[test]
    fn get_wval_limb_example_2() {
        // this is [128u8, 15u8, 3u8, 253u8] written in binary
        let scalar = Scalar256 {
            data: [0b11111101000000110000111110000000u64, 0u64, 0, 0],
        };
        let limb_1 = get_wval_limb(&scalar, 0, 6);
        assert_eq!(limb_1 & 0b00111111, 0b00000000);
        let limb_2 = get_wval_limb(&scalar, 6, 6);
        assert_eq!(limb_2 & 0b00111111, 0b00111110);
        let limb_3 = get_wval_limb(&scalar, 12, 6);
        assert_eq!(limb_3 & 0b00111111, 0b00110000);
        let limb_4 = get_wval_limb(&scalar, 18, 6);
        assert_eq!(limb_4 & 0b00111111, 0b00000000);
        let limb_5 = get_wval_limb(&scalar, 24, 6);
        assert_eq!(limb_5 & 0b00111111, 0b00111101);
        let limb_r = get_wval_limb(&scalar, 28, 8 % 6);
        assert_eq!(limb_r & 0b00000011, 0b00000011);
    }
}
