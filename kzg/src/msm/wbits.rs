//! Wbits MSM Implementation
//!
//! This algorithm is taken from https://github.com/crate-crypto/rust-eth-kzg
//!
//! Provides a fixed-base MSM using window-based precomputed tables with Booth encoding.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::{marker::PhantomData, ops::Neg};

use crate::{
    msm::FixedBaseMSM,
    Fr, G1Affine, G1Fp, G1GetFp, G1Mul, G1ProjAddAffine, G1,
};

#[cfg(feature = "diskcache")]
use crate::msm::diskcache::DiskCache;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for Wbits MSM.
#[derive(Debug, Clone, Copy)]
pub struct WbitsConfig {
    /// Override window size. If None, uses WINDOW_SIZE env var or default (8).
    pub window_size: Option<usize>,
}

impl Default for WbitsConfig {
    fn default() -> Self {
        Self { window_size: None }
    }
}

/// Get the window size from config or environment.
fn get_window_size_from_config(config: &WbitsConfig) -> usize {
    if let Some(ws) = config.window_size {
        return ws;
    }
    get_window_size_env()
}

fn get_window_size_env() -> usize {
    option_env!("WINDOW_SIZE")
        .map(|v| {
            v.parse()
                .expect("WINDOW_SIZE environment variable must be valid number")
        })
        .unwrap_or(8)
}

/// For backward compatibility with code that calls get_window_size()
fn get_window_size() -> usize {
    get_window_size_env()
}

// ============================================================================
// WBITS TABLE
// ============================================================================

#[derive(Debug, Clone)]
pub struct WbitsTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
where
    TFr: Fr,
    TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
{
    window_size: usize,
    numpoints: usize,
    points: Vec<TG1Affine>,

    // Legacy: batch tables for backward compatibility (will be removed)
    batch_numpoints: usize,
    batch_points: Vec<Vec<TG1Affine>>,

    // Use fn() -> T pattern to avoid requiring Send/Sync bounds on phantom types
    _phantom: PhantomData<fn() -> (TG1, TG1Fp, TFr, TG1ProjAddAffine)>,
}

// Code was taken from: https://github.com/privacy-scaling-explorations/halo2curves/blob/b753a832e92d5c86c5c997327a9cf9de86a18851/src/msm.rs#L13
pub fn get_booth_index(window_index: usize, window_size: usize, el: &[u8]) -> i32 {
    // Booth encoding:
    // * step by `window` size
    // * slice by size of `window + 1``
    // * each window overlap by 1 bit
    // * append a zero bit to the least significant end
    // Indexing rule for example window size 3 where we slice by 4 bits:
    // `[0, +1, +1, +2, +2, +3, +3, +4, -4, -3, -3 -2, -2, -1, -1, 0]``
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

/// This is the threshold to which batching the inversions in affine
/// formula costs more than doing mixed addition.
const BATCH_INVERSE_THRESHOLD: usize = 16;

/// Chooses between point addition and point doubling based on the input points.
///
/// Note: This does not handle the case where p1 == -p2.
///
/// This case is unlikely for our usecase, and is not trivial
/// to handle.
#[inline(always)]
fn choose_add_or_double<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
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
/// Panics if any of the elements are zero
pub fn batch_inverse_scratch_pad<F: G1Fp>(v: &mut [F], scratchpad: &mut Vec<F>) {
    // Montgomery's Trick and Fast Implementation of Masked AES
    // Genelle, Prouff and Quisquater
    // Section 3.2
    // but with an optimization to multiply every element in the returned vector by coeff

    // Clear the scratchpad and ensure it has enough capacity
    scratchpad.clear();
    scratchpad.reserve(v.len());

    // First pass: compute [a, ab, abc, ...]
    let mut tmp = F::one();
    for f in v.iter() {
        tmp = tmp.mul_fp(f);
        scratchpad.push(tmp.clone());
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
///
/// This function handles both addition of distinct points and point doubling.
#[inline(always)]
fn point_add_double<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
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
        max_bucket_length = std::cmp::max(max_bucket_length, points.len());
    }

    // Compute the total number of "unit of work"
    // In the single batch addition case this is analogous to
    // the batch inversion threshold
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
    // TODO: total_amount_of_work does not seem to be changing performance that much
    while total_amount_of_work > BATCH_INVERSE_THRESHOLD {
        // For each point, we check if they are odd and pop off
        // one of the points
        for (points, sum) in multi_points.iter_mut().zip(sums.iter_mut()) {
            // Make the number of points even
            if points.len() % 2 != 0 {
                TG1ProjAddAffine::add_or_double_assign_affine(sum, &points.pop().unwrap());
            }
        }

        denominators.clear();

        // For each pair of points over all
        // vectors, we collect them and put them in the
        // inverse array
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
            // The latter half of the vector is now unused,
            // all results are stored in the former half.
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
// FIXEDBASEMSM TRAIT IMPLEMENTATION
// ============================================================================

impl<
        TFr: Fr,
        TG1Fp: G1Fp,
        TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
        TG1Affine: G1Affine<TG1, TG1Fp>,
        TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
    > FixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine>
    for WbitsTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
{
    type Config = WbitsConfig;

    fn new(config: Self::Config, bases: &[TG1]) -> Result<Self, String> {
        let window_size = get_window_size_from_config(&config);
        let mut table = Vec::new();

        table
            .try_reserve_exact(bases.len() * (1 << (window_size - 1)))
            .map_err(|_| "WBITS precomputation table is too large".to_string())?;

        for point in bases {
            let mut current = point.clone();

            for _ in 0..(1 << (window_size - 1)) {
                table.push(TG1Affine::into_affine(&current));
                current = current.add_or_dbl(point);
            }
        }

        Ok(Self {
            window_size,
            numpoints: bases.len(),
            points: table,
            batch_numpoints: 0,
            batch_points: Vec::new(),
            _phantom: PhantomData,
        })
    }

    fn multiply(&self, scalars: &[TFr]) -> TG1 {
        // Wbits doesn't have a separate parallel implementation
        self.multiply_sequential(scalars)
    }

    fn multiply_sequential(&self, scalars: &[TFr]) -> TG1 {
        Self::multiply_sequential_raw(&self.points, scalars, self.window_size)
    }
}

// ============================================================================
// INTERNAL IMPLEMENTATION
// ============================================================================

impl<
        TFr: Fr,
        TG1Fp: G1Fp,
        TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
        TG1Affine: G1Affine<TG1, TG1Fp>,
        TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
    > WbitsTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
{
    fn multiply_sequential_raw(bases: &[TG1Affine], scalars: &[TFr], window_size: usize) -> TG1 {
        let scalars = scalars.iter().map(TFr::to_scalar).collect::<Vec<_>>();

        let number_of_windows = 255 / window_size + 1;
        let mut windows_of_points = vec![Vec::with_capacity(scalars.len()); number_of_windows];

        for window_idx in 0..windows_of_points.len() {
            for (scalar_idx, scalar_bytes) in scalars.iter().enumerate() {
                let sub_table = &bases[scalar_idx * (1 << (window_size - 1))
                    ..(scalar_idx + 1) * (1 << (window_size - 1))];

                let point_idx = get_booth_index(window_idx, window_size, scalar_bytes.as_u8());

                if point_idx == 0 {
                    continue;
                }
                let is_scalar_positive = point_idx.is_positive();
                let point_idx = point_idx.unsigned_abs() as usize - 1;
                let mut point = sub_table[point_idx];

                if !is_scalar_positive {
                    point = point.neg();
                }

                windows_of_points[window_idx].push(point);
            }
        }

        let accumulated_points =
            multi_batch_addition_binary_tree_stride::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(
                windows_of_points,
            );

        // Now accumulate the windows by doubling wbits times
        let mut result: TG1 = accumulated_points.last().unwrap().clone();
        for point in accumulated_points.into_iter().rev().skip(1) {
            // Double the result 'wbits' times
            for _ in 0..window_size {
                result = result.dbl();
            }
            // Add the accumulated point for this window
            result.add_or_dbl_assign(&point);
        }

        result
    }
}

// ============================================================================
// LEGACY COMPATIBILITY
// These methods maintain backward compatibility during the transition period.
// They will be removed after all callers are updated to use the new trait API.
// ============================================================================

impl<
        TFr: Fr,
        TG1Fp: G1Fp,
        TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
        TG1Affine: G1Affine<TG1, TG1Fp>,
        TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
    > WbitsTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
{
    fn try_read_cache(points: &[TG1], matrix: &[Vec<TG1>]) -> Result<Self, Option<[u8; 32]>> {
        #[cfg(feature = "diskcache")]
        {
            let window_size = get_window_size();
            DiskCache::<TG1, TG1Fp, TG1Affine>::load("wbits", window_size, points, matrix)
                .map_err(|(err, contenthash)| {
                    println!("Failed to load cache: {err}");
                    contenthash
                })
                .map(|cache| Self {
                    window_size,
                    numpoints: cache.numpoints,
                    points: cache.table,
                    batch_numpoints: cache.batch_numpoints,
                    batch_points: cache.batch_table,
                    _phantom: PhantomData,
                })
        }

        #[cfg(not(feature = "diskcache"))]
        Err(None)
    }

    fn try_write_cache(
        points: &[TG1],
        matrix: &[Vec<TG1>],
        table: &[TG1Affine],
        numpoints: usize,
        batch_table: &[Vec<TG1Affine>],
        batch_numpoints: usize,
        contenthash: Option<[u8; 32]>,
    ) -> Result<(), String> {
        #[cfg(feature = "diskcache")]
        {
            DiskCache::<TG1, TG1Fp, TG1Affine>::save(
                "wbits",
                get_window_size(),
                points,
                matrix,
                table,
                numpoints,
                batch_table,
                batch_numpoints,
                contenthash,
            )
            .inspect_err(|err| println!("Failed to save cache: {err}"))
        }

        #[cfg(not(feature = "diskcache"))]
        Ok(())
    }

    /// Legacy constructor for backward compatibility with precompute.rs.
    ///
    /// # Deprecated
    /// Use `FixedBaseMSM::new(config, bases)` for single-base MSM.
    #[allow(clippy::type_complexity)]
    pub fn new(points: &[TG1], matrix: &[Vec<TG1>]) -> Result<Option<Self>, String> {
        let window_size = get_window_size();
        let contenthash = match Self::try_read_cache(points, matrix) {
            Ok(v) => return Ok(Some(v)),
            Err(e) => e,
        };

        let mut table = Vec::new();

        table
            .try_reserve_exact(points.len() * (1 << (window_size - 1)))
            .map_err(|_| "WBITS precomputation table is too large".to_string())?;

        for point in points {
            let mut current = point.clone();

            for _ in 0..(1 << (window_size - 1)) {
                table.push(TG1Affine::into_affine(&current));
                current = current.add_or_dbl(point);
            }
        }

        if matrix.is_empty() {
            Self::try_write_cache(points, matrix, &table, points.len(), &[], 0, contenthash)?;
            Ok(Some(Self {
                window_size,
                numpoints: points.len(),
                points: table,
                batch_numpoints: 0,
                batch_points: Vec::new(),
                _phantom: PhantomData,
            }))
        } else {
            let batch_numpoints = matrix[0].len();

            let mut batch_points = Vec::new();
            batch_points
                .try_reserve_exact(matrix.len())
                .map_err(|_| "WBITS precomputation table is too large".to_owned())?;

            for row in matrix {
                let mut temp_table = Vec::new();
                temp_table
                    .try_reserve_exact(row.len() * (1 << (window_size - 1)))
                    .map_err(|_| "WBITS precomputation table is too large".to_owned())?;

                for point in row {
                    let mut current = point.clone();

                    for _ in 0..(1 << (window_size - 1)) {
                        temp_table.push(TG1Affine::into_affine(&current));
                        current = current.add_or_dbl(point);
                    }
                }

                batch_points.push(temp_table);
            }

            Self::try_write_cache(
                points,
                matrix,
                &table,
                points.len(),
                &batch_points,
                batch_numpoints,
                contenthash,
            )?;

            Ok(Some(Self {
                window_size,
                numpoints: points.len(),
                points: table,
                batch_numpoints,
                batch_points,
                _phantom: PhantomData,
            }))
        }
    }

    /// Legacy multiply_sequential - delegates to the trait method.
    ///
    /// Prefer importing `FixedBaseMSM` and using the trait method directly.
    pub fn multiply_sequential(&self, scalars: &[TFr]) -> TG1 {
        Self::multiply_sequential_raw(&self.points, scalars, self.window_size)
    }

    /// Legacy batch multiplication.
    ///
    /// For new code, use `BatchFixedBaseMSM` (coming in a future update).
    pub fn multiply_batch(&self, scalars: &[Vec<TFr>]) -> Vec<TG1> {
        assert!(scalars.len() == self.batch_points.len());
        let window_size = self.window_size;

        #[cfg(not(feature = "parallel"))]
        {
            self.batch_points
                .iter()
                .zip(scalars)
                .map(|(points, scalars)| Self::multiply_sequential_raw(points, scalars, window_size))
                .collect::<Vec<_>>()
        }

        #[cfg(feature = "parallel")]
        {
            use super::{
                cell::Cell,
                thread_pool::{da_pool, ThreadPoolExt},
            };
            use core::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::Arc;

            let pool = da_pool();
            let ncpus = pool.max_count();
            let counter = Arc::new(AtomicUsize::new(0));
            let mut results: Vec<Cell<TG1>> = Vec::with_capacity(scalars.len());
            #[allow(clippy::uninit_vec)]
            unsafe {
                results.set_len(results.capacity())
            };
            let results = &results[..];

            for _ in 0..ncpus {
                let counter = counter.clone();
                pool.joined_execute(move || loop {
                    let work = counter.fetch_add(1, Ordering::Relaxed);

                    if work >= scalars.len() {
                        break;
                    }

                    let result =
                        Self::multiply_sequential_raw(&self.batch_points[work], &scalars[work], window_size);
                    unsafe { *results[work].as_ptr().as_mut().unwrap() = result };
                });
            }

            pool.join();

            results.iter().map(|it| it.as_mut().clone()).collect()
        }
    }

    #[cfg(feature = "parallel")]
    pub fn multiply_parallel(&self, scalars: &[TFr]) -> TG1 {
        self.multiply_sequential(scalars)
    }
}
