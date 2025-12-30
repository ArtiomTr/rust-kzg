//! Pippenger MSM Implementation
//!
//! Variable-base Multi-Scalar Multiplication using the Pippenger/bucket method.
//! This module provides both sequential and parallel implementations.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{Fr, G1Affine, G1Fp, G1GetFp, Scalar256, G1};

use super::{
    utils::{
        booth_decode, booth_encode, get_wval_limb, is_zero, p1_dadd, p1_to_jacobian,
        pippenger_window_size, type_is_zero, type_zero, P1XYZZ,
    },
    VariableBaseMSM,
};

#[cfg(feature = "parallel")]
use super::utils::breakdown;

#[cfg(feature = "parallel")]
use super::{
    cell::Cell,
    thread_pool::{da_pool, ThreadPoolExt},
};

#[cfg(feature = "parallel")]
use alloc::sync::Arc;

#[cfg(feature = "parallel")]
use core::{
    num::Wrapping,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "parallel")]
use std::sync::{mpsc::channel, Barrier};

// ============================================================================
// PIPPENGER MSM STRUCT
// ============================================================================

/// Pippenger variable-base MSM implementation.
///
/// This struct implements `VariableBaseMSM` using the Pippenger bucket method.
/// Points are provided at multiply time, no precomputation is stored.
#[derive(Debug, Clone, Default)]
pub struct PippengerMSM<TFr, TG1, TG1Fp, TG1Affine> {
    _phantom: PhantomData<fn() -> (TFr, TG1, TG1Fp, TG1Affine)>,
}

impl<
        TFr: Fr,
        TG1: G1 + G1GetFp<TG1Fp>,
        TG1Fp: G1Fp,
        TG1Affine: G1Affine<TG1, TG1Fp>,
    > VariableBaseMSM<TFr, TG1, TG1Fp, TG1Affine> for PippengerMSM<TFr, TG1, TG1Fp, TG1Affine>
{
    fn multiply(&self, points: &[TG1], scalars: &[TFr]) -> TG1 {
        let points = batch_convert::<TG1, TG1Fp, TG1Affine>(points);
        let scalars: Vec<Scalar256> = scalars.iter().map(|s| s.to_scalar()).collect();

        #[cfg(feature = "parallel")]
        {
            tiling_parallel_pippenger(&points, &scalars)
        }

        #[cfg(not(feature = "parallel"))]
        {
            tiling_pippenger(&points, &scalars)
        }
    }

    fn multiply_sequential(&self, points: &[TG1], scalars: &[TFr]) -> TG1 {
        let points = TG1Affine::into_affines(points);
        let scalars: Vec<Scalar256> = scalars.iter().map(|s| s.to_scalar()).collect();
        tiling_pippenger(&points, &scalars)
    }
}

// ============================================================================
// BATCH CONVERSION
// ============================================================================

/// Batch convert projective points to affine.
///
/// Uses parallel conversion if `parallel` feature is enabled and size is large enough.
pub fn batch_convert<TG1: G1, TFp: G1Fp, TG1Affine: G1Affine<TG1, TFp> + Sized>(
    points: &[TG1],
) -> Vec<TG1Affine> {
    #[cfg(feature = "parallel")]
    return parallel_affine_conv::<TG1, TFp, TG1Affine>(points);

    #[cfg(not(feature = "parallel"))]
    return TG1Affine::into_affines(points);
}

/// Parallel affine conversion using thread pool.
#[cfg(feature = "parallel")]
pub fn parallel_affine_conv<TG1: G1, TFp: G1Fp, TG1Affine: G1Affine<TG1, TFp> + Sized>(
    points: &[TG1],
) -> Vec<TG1Affine> {
    let npoints = points.len();
    let pool = da_pool();
    let ncpus = pool.max_count();
    if ncpus < 2 || npoints < 768 {
        return TG1Affine::into_affines(points);
    }

    let mut ret = Vec::<TG1Affine>::with_capacity(npoints);
    #[allow(clippy::uninit_vec)]
    unsafe {
        ret.set_len(npoints)
    };

    let mut nslices = npoints.div_ceil(512);
    nslices = core::cmp::min(nslices, ncpus);
    let wg = Arc::new((Barrier::new(2), AtomicUsize::new(nslices)));

    let (mut delta, mut rem) = (npoints / nslices + 1, Wrapping(npoints % nslices));
    let mut x = 0usize;
    while x < npoints {
        delta -= (rem == Wrapping(0)) as usize;
        rem -= Wrapping(1);

        let out = &mut ret[x..x + delta];
        let inp = &points[x..x + delta];

        x += delta;

        let wg = wg.clone();
        pool.joined_execute(move || {
            TG1Affine::into_affines_loc(out, inp);
            if wg.1.fetch_sub(1, Ordering::AcqRel) == 1 {
                wg.0.wait();
            }
        });
    }
    wg.0.wait();

    ret
}

// ============================================================================
// BUCKET INTEGRATION
// ============================================================================

/// Calculate bucket sum.
///
/// Multiplies the point in each bucket by its index, then sums all results.
///
/// # Arguments
/// * `out` - output where bucket sum is written
/// * `buckets` - array of buckets
/// * `wbits` - window size (q^window)
pub fn p1_integrate_buckets<TG1: G1 + G1GetFp<TFp>, TFp: G1Fp>(
    out: &mut TG1,
    buckets: &mut [P1XYZZ<TFp>],
    wbits: usize,
) {
    let mut n = (1usize << wbits) - 1;
    let mut ret = buckets[n];
    let mut acc = buckets[n];

    type_zero(&mut buckets[n]);
    loop {
        if n == 0 {
            break;
        }
        n -= 1;

        if type_is_zero(&buckets[n]) == 0 {
            p1_dadd(&mut acc, &buckets[n]);
            type_zero(&mut buckets[n]);
        }
        p1_dadd(&mut ret, &acc);
    }

    p1_to_jacobian(out, &ret);
}

// ============================================================================
// TILE PROCESSING
// ============================================================================

/// Process a tile in the Pippenger algorithm (public interface).
#[allow(clippy::too_many_arguments)]
pub fn p1s_tile_pippenger_pub<TG1: G1 + G1GetFp<TFp>, TFp: G1Fp, TG1Affine: G1Affine<TG1, TFp>>(
    ret: &mut TG1,
    points: &[TG1Affine],
    scalars: &[Scalar256],
    buckets: &mut [P1XYZZ<TFp>],
    bit0: usize,
    window: usize,
) {
    const NBITS: usize = 255;
    let (wbits, cbits) = if bit0 + window > NBITS {
        let wbits = NBITS - bit0;
        (wbits, wbits + 1)
    } else {
        (window, window)
    };

    p1s_tile_pippenger(ret, points, scalars, buckets, bit0, wbits, cbits);
}

/// Process a tile in the Pippenger algorithm (internal).
#[allow(clippy::too_many_arguments)]
pub fn p1s_tile_pippenger<TG1: G1 + G1GetFp<TFp>, TFp: G1Fp, TG1Affine: G1Affine<TG1, TFp>>(
    ret: &mut TG1,
    points: &[TG1Affine],
    scalars: &[Scalar256],
    buckets: &mut [P1XYZZ<TFp>],
    bit0: usize,
    wbits: usize,
    cbits: usize,
) {
    // Create mask with `wbits` ones at the end
    let wmask = (1u64 << (wbits + 1)) - 1;

    // Check if bit0 is zero
    let z = is_zero(bit0.try_into().unwrap());

    // Offset bit0 by 1 if not zero
    let bit0 = bit0 - (z ^ 1) as usize;

    // Increase wbits by one if bit0 was not zero
    let wbits = wbits + (z ^ 1) as usize;

    for (point, scalar) in points.iter().zip(scalars.iter()) {
        // Calculate window value (encoded bucket index)
        let wval = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
        let wval = booth_encode(wval, cbits);

        // Move point to corresponding bucket
        booth_decode(buckets, wval, cbits, point);
    }

    // Integrate buckets
    p1_integrate_buckets(ret, buckets, cbits - 1);
}

// ============================================================================
// SEQUENTIAL PIPPENGER
// ============================================================================

/// Sequential Pippenger MSM implementation.
pub fn tiling_pippenger<TG1: G1 + G1GetFp<TG1Fp>, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    points: &[TG1Affine],
    scalars: &[Scalar256],
) -> TG1 {
    let window = pippenger_window_size(points.len());
    let mut buckets = vec![P1XYZZ::<TG1Fp>::default(); 1 << (window - 1)];

    let mut wbits: usize = 255 % window;
    let mut cbits: usize = wbits + 1;
    let mut bit0: usize = 255;
    let mut tile = TG1::zero();

    let mut ret = TG1::zero();

    loop {
        bit0 -= wbits;
        if bit0 == 0 {
            break;
        }

        p1s_tile_pippenger(&mut tile, points, scalars, &mut buckets, bit0, wbits, cbits);

        ret.add_or_dbl_assign(&tile);
        for _ in 0..window {
            ret.dbl_assign();
        }
        cbits = window;
        wbits = window;
    }
    p1s_tile_pippenger(&mut tile, points, scalars, &mut buckets, 0, wbits, cbits);
    ret.add_or_dbl_assign(&tile);
    ret
}

// ============================================================================
// PARALLEL PIPPENGER
// ============================================================================

#[cfg(feature = "parallel")]
struct Tile {
    x: usize,
    dx: usize,
    y: usize,
    dy: usize,
}

/// Parallel Pippenger MSM implementation using thread pool.
#[cfg(feature = "parallel")]
pub fn tiling_parallel_pippenger<
    TG1: G1 + G1GetFp<TG1Fp>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
>(
    mut points: &[TG1Affine],
    scalars: &[Scalar256],
) -> TG1 {
    if scalars.len() < points.len() {
        points = &points[0..scalars.len()];
    }
    let npoints = points.len();

    let pool = da_pool();
    let ncpus = pool.max_count();

    if ncpus < 2 || npoints < 32 {
        return tiling_pippenger(points, scalars);
    }

    let (nx, ny, window) = breakdown(pippenger_window_size(npoints), ncpus);

    // grid[] holds "coordinates" and place for result
    let mut grid: Vec<(Tile, Cell<TG1>)> = Vec::with_capacity(nx * ny);
    #[allow(clippy::uninit_vec)]
    unsafe {
        grid.set_len(grid.capacity())
    };
    let dx = npoints / nx;
    let mut y = window * (ny - 1);
    let mut total = 0usize;

    while total < nx {
        grid[total].0.x = total * dx;
        grid[total].0.dx = dx;
        grid[total].0.y = y;
        grid[total].0.dy = 255 - y;
        total += 1;
    }
    grid[total - 1].0.dx = npoints - grid[total - 1].0.x;
    while y != 0 {
        y -= window;
        for i in 0..nx {
            grid[total].0.x = grid[i].0.x;
            grid[total].0.dx = grid[i].0.dx;
            grid[total].0.y = y;
            grid[total].0.dy = window;
            total += 1;
        }
    }
    let grid = &grid[..];

    let points = points;

    let mut row_sync: Vec<AtomicUsize> = Vec::with_capacity(ny);
    row_sync.resize_with(ny, Default::default);
    let row_sync = Arc::new(row_sync);
    let counter = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = channel();
    let n_workers = core::cmp::min(ncpus, total);
    for _ in 0..n_workers {
        let tx = tx.clone();
        let counter = counter.clone();
        let row_sync = row_sync.clone();

        pool.joined_execute(move || {
            let mut buckets = vec![P1XYZZ::<TG1Fp>::default(); 1 << (window - 1)];
            loop {
                let work = counter.fetch_add(1, Ordering::Relaxed);
                if work >= total {
                    break;
                }

                let x = grid[work].0.x;
                let y = grid[work].0.y;
                let dx = grid[work].0.dx;

                p1s_tile_pippenger_pub(
                    grid[work].1.as_mut(),
                    &points[x..(x + dx)],
                    &scalars[x..],
                    &mut buckets,
                    y,
                    window,
                );
                if row_sync[y / window].fetch_add(1, Ordering::AcqRel) == nx - 1 {
                    tx.send(y).expect("disaster");
                }
            }
        });
    }

    let mut ret = <TG1>::default();
    let mut rows = vec![false; ny];
    let mut row = 0usize;
    for _ in 0..ny {
        let mut y = rx.recv().unwrap();
        rows[y / window] = true;
        while grid[row].0.y == y {
            while row < total && grid[row].0.y == y {
                ret.add_or_dbl_assign(grid[row].1.as_mut());
                row += 1;
            }
            if y == 0 {
                break;
            }
            for _ in 0..window {
                ret.dbl_assign();
            }
            y -= window;
            if !rows[y / window] {
                break;
            }
        }
    }
    ret
}
