//! BGMW (Bos-Coster Multi-Window) MSM Implementation
//!
//! This module provides a fixed-base MSM algorithm using precomputed tables.
//! The algorithm decomposes scalars into q-ary representation and uses
//! bucket accumulation with Booth encoding.

use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    msm::{
        pippenger::p1_integrate_buckets,
        utils::{booth_decode, booth_encode, get_wval_limb, is_zero, num_bits, P1XYZZ},
        FixedBaseMSM,
    },
    Fr, G1Affine, G1Fp, G1GetFp, G1Mul, G1ProjAddAffine, Scalar256, G1,
};

const NBITS: usize = 255;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for BGMW MSM.
#[derive(Debug, Clone, Copy)]
pub struct BgmwConfig {
    /// Override window size. If None, automatically calculated based on point count.
    pub window_size: Option<usize>,
}

impl Default for BgmwConfig {
    fn default() -> Self {
        Self { window_size: None }
    }
}

// ============================================================================
// WINDOW HANDLING
// ============================================================================

#[cfg(feature = "parallel")]
#[derive(Debug, Clone, Copy)]
enum BgmwWindow {
    Sync(usize),
    Parallel((usize, usize, usize)),
}

#[cfg(not(feature = "parallel"))]
type BgmwWindow = usize;

#[inline]
const fn get_table_dimensions(window: BgmwWindow) -> (usize, usize) {
    let window_width;

    #[cfg(not(feature = "parallel"))]
    {
        window_width = window;
    }

    #[cfg(feature = "parallel")]
    {
        window_width = match window {
            BgmwWindow::Sync(wnd) => wnd,
            BgmwWindow::Parallel((_, ny, wnd)) => return (wnd, ny),
        }
    }

    let h = NBITS.div_ceil(window_width) + is_zero((NBITS % window_width) as u64) as usize;

    (window_width, h)
}

#[inline]
const fn get_sequential_window_size(window: BgmwWindow) -> usize {
    #[cfg(not(feature = "parallel"))]
    {
        window
    }

    #[cfg(feature = "parallel")]
    {
        match window {
            BgmwWindow::Sync(wnd) => wnd,
            BgmwWindow::Parallel(_) => {
                panic!("Cannot use parallel BGMW table in sequential version")
            }
        }
    }
}

/// Calculate optimal window size for BGMW.
///
/// Approximates minimum of: y = ceil(255/w) * npoints + 2^w - 2
fn bgmw_window_size(npoints: usize, config: &BgmwConfig) -> usize {
    if let Some(window) = config.window_size {
        return window;
    }

    option_env!("WINDOW_SIZE")
        .map(|v| {
            v.parse()
                .expect("WINDOW_SIZE environment variable must be valid number")
        })
        .unwrap_or({
            let wbits = num_bits(npoints);

            match wbits {
                1 => 4,
                2..=3 => 5,
                4 => 6,
                5 => 7,
                6..=7 => 8,
                8 => 9,
                9..=10 => 10,
                11 => 11,
                12 => 12,
                13..=14 => 13,
                15..=16 => 15,
                17 => 16,
                18..=19 => 17,
                20 => 19,
                21..=22 => 20,
                23..=24 => 22,
                25..=26 => 24,
                27..=29 => 26,
                30..=32 => 29,
                33..=37 => 32,
                _ => 37,
            }
        })
}

#[cfg(feature = "parallel")]
#[allow(clippy::option_env_unwrap)]
fn bgmw_parallel_window_size(
    npoints: usize,
    ncpus: usize,
    config: &BgmwConfig,
) -> (usize, usize, usize) {
    option_env!("WINDOW_NX")
        .and_then(|v| v.parse().ok())
        .map(|nx| {
            let wnd = config.window_size.unwrap_or_else(|| {
                option_env!("WINDOW_SIZE")
                    .expect(
                        "Unable to use BGMW: when specifying WINDOW_NX environment \
                        variable, please also specify WINDOW_SIZE",
                    )
                    .parse()
                    .expect("WINDOW_SIZE environment variable must be valid number")
            });

            (
                nx,
                255usize.div_ceil(wnd) + is_zero((NBITS % wnd) as u64) as usize,
                wnd,
            )
        })
        .unwrap_or({
            let mut min_ops = usize::MAX;
            let mut opt = 0;

            let mut win = 2;
            while win <= 40 {
                let ops = (1 << win) + (255usize.div_ceil(win).div_ceil(ncpus) * npoints) - 2;
                if min_ops >= ops {
                    min_ops = ops;
                    opt = win;
                }
                win += 1;
            }

            let mut mult = 1;
            let mut opt_x = 1;

            while mult <= 8 {
                let nx = ncpus * mult;
                let wnd = bgmw_window_size(npoints / nx, config);

                let ops = mult * 255usize.div_ceil(wnd) * npoints.div_ceil(nx) + (1 << wnd) - 2;

                if min_ops > ops {
                    min_ops = ops;
                    opt = wnd;
                    opt_x = nx;
                }

                mult += 1;
            }

            (
                opt_x,
                255usize.div_ceil(opt) + is_zero((NBITS % opt) as u64) as usize,
                opt,
            )
        })
}

// ============================================================================
// BGMW TABLE
// ============================================================================

/// BGMW precomputation table for fixed-base MSM.
#[derive(Debug, Clone)]
pub struct BgmwTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
where
    TFr: Fr,
    TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
{
    window: BgmwWindow,
    points: Vec<TG1Affine>,
    numpoints: usize,
    h: usize,

    // Legacy: batch tables for backward compatibility (will be removed)
    batch_tables: Option<Vec<(Vec<TG1Affine>, usize, usize, BgmwWindow)>>,

    // Use fn() -> T pattern to avoid requiring Send/Sync bounds on phantom types
    _phantom: PhantomData<fn() -> (TG1, TG1Fp, TFr, TG1ProjAddAffine)>,
}

impl<
        TFr: Fr,
        TG1Fp: G1Fp,
        TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
        TG1Affine: G1Affine<TG1, TG1Fp>,
        TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
    > FixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine>
    for BgmwTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
{
    type Config = BgmwConfig;

    fn new(config: Self::Config, bases: &[TG1]) -> Result<Self, alloc::string::String> {
        let window = Self::compute_window(bases.len(), &config);
        let (window_width, h) = get_table_dimensions(window);

        let mut table: Vec<TG1Affine> = Vec::new();
        let q = TFr::from_u64(1u64 << window_width);

        table
            .try_reserve_exact(bases.len() * h)
            .map_err(|_| "BGMW precomputation table is too large".to_string())?;

        unsafe { table.set_len(bases.len() * h) };

        for i in 0..bases.len() {
            let mut tmp_point = bases[i].clone();
            for j in 0..h {
                let idx = j * bases.len() + i;
                table[idx] = TG1Affine::into_affine(&tmp_point);
                tmp_point = tmp_point.mul(&q);
            }
        }

        Ok(Self {
            numpoints: bases.len(),
            points: table,
            window,
            h,
            batch_tables: None,
            _phantom: PhantomData,
        })
    }

    fn multiply(&self, scalars: &[TFr]) -> TG1 {
        #[cfg(feature = "parallel")]
        {
            if let BgmwWindow::Parallel(_) = self.window {
                return self.multiply_parallel_impl(scalars);
            }
        }
        self.multiply_sequential(scalars)
    }

    fn multiply_sequential(&self, scalars: &[TFr]) -> TG1 {
        let window = get_sequential_window_size(self.window);
        let mut buckets = vec![P1XYZZ::<TG1Fp>::default(); 1 << (window - 1)];

        Self::multiply_sequential_raw(
            &self.points,
            scalars,
            &mut buckets,
            window,
            self.numpoints,
            self.h,
        )
    }
}

impl<
        TFr: Fr,
        TG1Fp: G1Fp,
        TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
        TG1Affine: G1Affine<TG1, TG1Fp>,
        TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
    > BgmwTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
{
    fn compute_window(npoints: usize, config: &BgmwConfig) -> BgmwWindow {
        #[cfg(feature = "parallel")]
        {
            use super::thread_pool::da_pool;

            let pool = da_pool();
            let ncpus = pool.max_count();

            if npoints >= 32 && ncpus >= 2 {
                BgmwWindow::Parallel(bgmw_parallel_window_size(npoints, ncpus, config))
            } else {
                BgmwWindow::Sync(bgmw_window_size(npoints, config))
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            bgmw_window_size(npoints, config)
        }
    }

    fn multiply_sequential_raw(
        points: &[TG1Affine],
        scalars: &[TFr],
        buckets: &mut [P1XYZZ<TG1Fp>],
        window: usize,
        numpoints: usize,
        h: usize,
    ) -> TG1 {
        let scalars = scalars.iter().map(TFr::to_scalar).collect::<Vec<_>>();
        let scalars = &scalars[..];

        let mut wbits: usize = 255 % window;
        let mut cbits: usize = wbits + 1;
        let mut bit0: usize = 255;

        let mut q_idx = h;

        loop {
            bit0 -= wbits;
            q_idx -= 1;
            if bit0 == 0 {
                break;
            }

            p1_tile_bgmw(
                &points[q_idx * numpoints..(q_idx + 1) * numpoints],
                scalars,
                buckets,
                bit0,
                wbits,
                cbits,
            );

            cbits = window;
            wbits = window;
        }
        p1_tile_bgmw(&points[0..numpoints], scalars, buckets, 0, wbits, cbits);

        let mut ret = TG1::default();
        p1_integrate_buckets(&mut ret, buckets, wbits - 1);

        ret
    }

    #[cfg(feature = "parallel")]
    fn multiply_parallel_impl(&self, scalars: &[TFr]) -> TG1 {
        use super::{
            cell::Cell,
            pippenger::tiling_pippenger,
            thread_pool::{da_pool, ThreadPoolExt},
        };
        use core::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{mpsc, Arc};

        let npoints = scalars.len();
        let pool = da_pool();
        let ncpus = pool.max_count();

        if ncpus > npoints || npoints < 32 {
            let scalars = scalars.iter().map(TFr::to_scalar).collect::<Vec<_>>();
            return tiling_pippenger(&self.points[0..npoints], &scalars);
        }

        struct Tile {
            x: usize,
            dx: usize,
            y: usize,
            dy: usize,
        }

        let (nx, ny, window) = match self.window {
            BgmwWindow::Sync(_) => return self.multiply_sequential(scalars),
            BgmwWindow::Parallel(values) => values,
        };

        let scalars = scalars.iter().map(TFr::to_scalar).collect::<Vec<_>>();
        let scalars = &scalars[..];

        // |grid[]| holds "coordinates"
        let mut grid: Vec<Tile> = Vec::with_capacity(nx * ny);
        #[allow(clippy::uninit_vec)]
        unsafe {
            grid.set_len(grid.capacity())
        };
        let dx = npoints / nx;
        let mut y = window * (ny - 1);
        let mut total = 0usize;

        while total < nx {
            grid[total].x = total * dx;
            grid[total].dx = dx;
            grid[total].y = y;
            grid[total].dy = NBITS - y;
            total += 1;
        }
        grid[total - 1].dx = npoints - grid[total - 1].x;
        while y != 0 {
            y -= window;
            for i in 0..nx {
                grid[total].x = grid[i].x;
                grid[total].dx = grid[i].dx;
                grid[total].y = y;
                grid[total].dy = window;
                total += 1;
            }
        }
        let grid = &grid[..];

        let mut row_sync: Vec<AtomicUsize> = Vec::with_capacity(ny);
        row_sync.resize_with(ny, Default::default);
        let counter = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel();
        let n_workers = core::cmp::min(ncpus, total);

        let mut results: Vec<Cell<TG1>> = Vec::with_capacity(n_workers);
        #[allow(clippy::uninit_vec)]
        unsafe {
            results.set_len(results.capacity());
        };

        let results = &results[..];

        #[allow(clippy::needless_range_loop)]
        for worker_index in 0..n_workers {
            let tx = tx.clone();
            let counter = counter.clone();

            pool.joined_execute(move || {
                let mut buckets = vec![P1XYZZ::<TG1Fp>::default(); 1 << (window - 1)];
                loop {
                    let work = counter.fetch_add(1, Ordering::Relaxed);
                    if work >= total {
                        p1_integrate_buckets(
                            unsafe { results[worker_index].as_ptr().as_mut() }.unwrap(),
                            &mut buckets,
                            window - 1,
                        );
                        tx.send(worker_index).expect("disaster");

                        break;
                    }

                    let x = grid[work].x;
                    let y = grid[work].y;
                    let dx = grid[work].dx;

                    let row_start = (y / window) * self.numpoints + x;
                    let points = &self.points[row_start..(row_start + dx)];

                    let (wbits, cbits) = if y + window > NBITS {
                        let wbits = NBITS - y;
                        (wbits, wbits + 1)
                    } else {
                        (window, window)
                    };

                    p1_tile_bgmw(points, &scalars[x..(x + dx)], &mut buckets, y, wbits, cbits);
                }
            });
        }

        let mut ret = TG1::zero();
        for _ in 0..n_workers {
            let idx = rx.recv().unwrap();
            ret.add_or_dbl_assign(results[idx].as_mut());
        }
        ret
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
    > BgmwTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
{
    /// Legacy constructor for backward compatibility with precompute.rs.
    ///
    /// Creates a precomputation table from points (single base) and matrix (batch bases).
    ///
    /// # Deprecated
    /// Use `FixedBaseMSM::new(config, bases)` for single-base MSM.
    /// For batch operations, BatchFixedBaseMSM will be available in a future update.
    #[allow(clippy::type_complexity)]
    pub fn new(
        points: &[TG1],
        matrix: &[Vec<TG1>],
    ) -> Result<Option<Self>, alloc::string::String> {
        let config = BgmwConfig::default();

        // Create main table for single base
        let window = Self::compute_window(points.len(), &config);
        let (window_width, h) = get_table_dimensions(window);

        let mut table: Vec<TG1Affine> = Vec::new();
        let q = TFr::from_u64(1u64 << window_width);

        table
            .try_reserve_exact(points.len() * h)
            .map_err(|_| "BGMW precomputation table is too large".to_string())?;

        unsafe { table.set_len(points.len() * h) };

        for i in 0..points.len() {
            let mut tmp_point = points[i].clone();
            for j in 0..h {
                let idx = j * points.len() + i;
                table[idx] = TG1Affine::into_affine(&tmp_point);
                tmp_point = tmp_point.mul(&q);
            }
        }

        // Create batch tables if matrix is non-empty
        let batch_tables = if matrix.is_empty() {
            None
        } else {
            let mut batch = Vec::with_capacity(matrix.len());
            for row in matrix {
                let batch_window = Self::compute_window(row.len(), &config);
                let (batch_window_width, batch_h) = get_table_dimensions(batch_window);

                let mut batch_table: Vec<TG1Affine> = Vec::new();
                let batch_q = TFr::from_u64(1u64 << batch_window_width);

                batch_table
                    .try_reserve_exact(row.len() * batch_h)
                    .map_err(|_| "BGMW batch precomputation table is too large".to_string())?;

                unsafe { batch_table.set_len(row.len() * batch_h) };

                for i in 0..row.len() {
                    let mut tmp_point = row[i].clone();
                    for j in 0..batch_h {
                        let idx = j * row.len() + i;
                        batch_table[idx] = TG1Affine::into_affine(&tmp_point);
                        tmp_point = tmp_point.mul(&batch_q);
                    }
                }

                batch.push((batch_table, row.len(), batch_h, batch_window));
            }
            Some(batch)
        };

        Ok(Some(Self {
            numpoints: points.len(),
            points: table,
            window,
            h,
            batch_tables,
            _phantom: PhantomData,
        }))
    }

    /// Legacy multiply_sequential - delegates to trait method.
    ///
    /// Prefer importing `FixedBaseMSM` and using the trait method directly.
    pub fn multiply_sequential(&self, scalars: &[TFr]) -> TG1 {
        <Self as FixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine>>::multiply_sequential(self, scalars)
    }

    /// Legacy multiply_parallel - delegates to multiply method.
    ///
    /// Prefer importing `FixedBaseMSM` and using `multiply()` directly.
    #[cfg(feature = "parallel")]
    pub fn multiply_parallel(&self, scalars: &[TFr]) -> TG1 {
        <Self as FixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine>>::multiply(self, scalars)
    }

    /// Legacy batch multiplication.
    ///
    /// For new code, use `BatchFixedBaseMSM` (coming in a future update).
    pub fn multiply_batch(&self, scalars: &[Vec<TFr>]) -> Vec<TG1> {
        let batch_tables = self
            .batch_tables
            .as_ref()
            .expect("multiply_batch called but no batch tables were created");

        assert_eq!(
            batch_tables.len(),
            scalars.len(),
            "Batch size mismatch: expected {} rows, got {}",
            batch_tables.len(),
            scalars.len()
        );

        #[cfg(feature = "parallel")]
        {
            use super::thread_pool::{da_pool, ThreadPoolExt};
            use core::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::{mpsc, Arc};

            use super::cell::Cell;

            let pool = da_pool();
            let ncpus = pool.max_count();
            let total = scalars.len();

            if ncpus <= 1 || total < 2 {
                return self.multiply_batch_sequential(scalars, batch_tables);
            }

            let counter = Arc::new(AtomicUsize::new(0));
            let (tx, rx) = mpsc::channel();
            let n_workers = core::cmp::min(ncpus, total);

            let mut results: Vec<Cell<TG1>> = Vec::with_capacity(total);
            #[allow(clippy::uninit_vec)]
            unsafe {
                results.set_len(results.capacity());
            }

            let results = &results[..];

            for _ in 0..n_workers {
                let tx = tx.clone();
                let counter = counter.clone();

                pool.joined_execute(move || {
                    loop {
                        let work = counter.fetch_add(1, Ordering::Relaxed);
                        if work >= total {
                            break;
                        }

                        let (ref batch_points, numpoints, h, batch_window) = batch_tables[work];
                        let window = get_sequential_window_size(batch_window);
                        let mut buckets = vec![P1XYZZ::<TG1Fp>::default(); 1 << (window - 1)];

                        let result = Self::multiply_sequential_raw(
                            batch_points,
                            &scalars[work],
                            &mut buckets,
                            window,
                            numpoints,
                            h,
                        );

                        unsafe {
                            *results[work].as_ptr().as_mut().unwrap() = result;
                        }
                    }
                    tx.send(()).expect("Failed to send completion signal");
                });
            }

            for _ in 0..n_workers {
                rx.recv().unwrap();
            }

            results.iter().map(|c| c.as_mut().clone()).collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.multiply_batch_sequential(scalars, batch_tables)
        }
    }

    fn multiply_batch_sequential(
        &self,
        scalars: &[Vec<TFr>],
        batch_tables: &[(Vec<TG1Affine>, usize, usize, BgmwWindow)],
    ) -> Vec<TG1> {
        batch_tables
            .iter()
            .zip(scalars.iter())
            .map(|((batch_points, numpoints, h, batch_window), scalars)| {
                let window = get_sequential_window_size(*batch_window);
                let mut buckets = vec![P1XYZZ::<TG1Fp>::default(); 1 << (window - 1)];
                Self::multiply_sequential_raw(batch_points, scalars, &mut buckets, window, *numpoints, *h)
            })
            .collect()
    }
}

// ============================================================================
// TILE PROCESSING
// ============================================================================

/// Process a tile in the BGMW algorithm.
///
/// Moves points to buckets based on their scalar window values using Booth encoding.
#[allow(clippy::too_many_arguments)]
pub fn p1_tile_bgmw<TG1: G1 + G1GetFp<TFp>, TFp: G1Fp, TG1Affine: G1Affine<TG1, TFp>>(
    points: &[TG1Affine],
    scalars: &[Scalar256],
    buckets: &mut [P1XYZZ<TFp>],
    bit0: usize,
    wbits: usize,
    cbits: usize,
) {
    if scalars.is_empty() {
        return;
    }

    // Get first scalar
    let scalar = &scalars[0];

    // Get first point
    let point = &points[0];

    // Create mask, that contains `wbits` ones at the end.
    let wmask = (1u64 << (wbits + 1)) - 1;

    // Check if `bit0` is zero. `z` is set to `1` when `bit0 = 0`, and `0` otherwise.
    let z = is_zero(bit0.try_into().unwrap());

    // Offset `bit0` by 1, if it is not equal to zero.
    let bit0 = bit0 - (z ^ 1) as usize;

    // Increase `wbits` by one, if `bit0` is not equal to zero.
    let wbits = wbits + (z ^ 1) as usize;

    // Calculate first window value (encoded bucket index)
    let wval = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let mut wval = booth_encode(wval, cbits);

    if scalars.len() == 1 {
        booth_decode(buckets, wval, cbits, point);
        return;
    }

    // Get second scalar
    let scalar = &scalars[1];

    // Calculate second window value (encoded bucket index)
    let wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let mut wnxt = booth_encode(wnxt, cbits);

    // Move first point to corresponding bucket
    booth_decode(buckets, wval, cbits, point);

    // Last point will be calculated separately, so decrementing point count
    let npoints = scalars.len() - 1;

    // Move points to buckets
    for i in 1..npoints {
        // Get current window value (encoded bucket index)
        wval = wnxt;

        // Get next scalar
        let scalar = &scalars[i + 1];
        // Get next window value (encoded bucket index)
        wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
        wnxt = booth_encode(wnxt, cbits);

        // Get current point
        let point = &points[i];

        // Move point to corresponding bucket (add or subtract from bucket)
        booth_decode(buckets, wval, cbits, point);
    }
    // Get last point
    let point = &points[npoints];
    // Move point to bucket
    booth_decode(buckets, wnxt, cbits, point);
}
