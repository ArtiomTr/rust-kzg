/// This algorithm is taken from https://github.com/crate-crypto/rust-eth-kzg
use core::{marker::PhantomData, ops::Neg};

use super::batch_addition::multi_batch_addition_binary_tree_stride;
#[cfg(feature = "diskcache")]
use crate::msm::diskcache::DiskCache;
use crate::{Fr, G1Affine, G1Fp, G1GetFp, G1Mul, G1ProjAddAffine, G1};

#[derive(Debug, Clone)]
pub struct WbitsTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
where
    TFr: Fr,
    TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
{
    numpoints: usize,
    points: Vec<TG1Affine>,

    batch_numpoints: usize,
    batch_points: Vec<Vec<TG1Affine>>,

    g1_marker: PhantomData<TG1>,
    g1_fp_marker: PhantomData<TG1Fp>,
    fr_marker: PhantomData<TFr>,
    g1_affine_add_marker: PhantomData<TG1ProjAddAffine>,
}

fn get_window_size() -> usize {
    option_env!("WINDOW_SIZE")
        .map(|v| {
            v.parse()
                .expect("WINDOW_SIZE environment variable must be valid number")
        })
        .unwrap_or(8)
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
            DiskCache::<TG1, TG1Fp, TG1Affine>::load("wbits", get_window_size(), points, matrix)
                .map_err(|(err, contenthash)| {
                    println!("Failed to load cache: {err}");
                    contenthash
                })
                .map(|cache| Self {
                    numpoints: cache.numpoints,
                    points: cache.table,
                    batch_numpoints: cache.batch_numpoints,
                    batch_points: cache.batch_table,

                    g1_marker: PhantomData,
                    g1_fp_marker: PhantomData,
                    fr_marker: PhantomData,
                    g1_affine_add_marker: PhantomData,
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

    pub fn new(points: &[TG1], matrix: &[Vec<TG1>]) -> Result<Option<Self>, String> {
        let contenthash = match Self::try_read_cache(points, matrix) {
            Ok(v) => return Ok(Some(v)),
            Err(e) => e,
        };

        let mut table = Vec::new();

        table
            .try_reserve_exact(points.len() * (1 << (get_window_size() - 1)))
            .map_err(|_| "WBITS precomputation table is too large".to_string())?;

        for point in points {
            let mut current = point.clone();

            for _ in 0..(1 << (get_window_size() - 1)) {
                table.push(TG1Affine::into_affine(&current));
                current = current.add_or_dbl(point);
            }
        }

        if matrix.is_empty() {
            Self::try_write_cache(points, matrix, &table, points.len(), &[], 0, contenthash)?;
            Ok(Some(Self {
                numpoints: points.len(),
                points: table,
                batch_numpoints: 0,
                batch_points: Vec::new(),

                g1_marker: PhantomData,
                g1_fp_marker: PhantomData,
                fr_marker: PhantomData,
                g1_affine_add_marker: PhantomData,
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
                    .try_reserve_exact(row.len() * (1 << (get_window_size() - 1)))
                    .map_err(|_| "WBITS precomputation table is too large".to_owned())?;

                for point in row {
                    let mut current = point.clone();

                    for _ in 0..(1 << (get_window_size() - 1)) {
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
                numpoints: points.len(),
                points: table,

                batch_numpoints,
                batch_points,

                fr_marker: PhantomData,
                g1_fp_marker: PhantomData,
                g1_marker: PhantomData,
                g1_affine_add_marker: PhantomData,
            }))
        }
    }

    fn multiply_sequential_raw(bases: &[TG1Affine], scalars: &[TFr]) -> TG1 {
        let scalars = scalars.iter().map(TFr::to_scalar).collect::<Vec<_>>();

        let number_of_windows = 255 / get_window_size() + 1;
        let mut windows_of_points = vec![Vec::with_capacity(scalars.len()); number_of_windows];

        for window_idx in 0..windows_of_points.len() {
            for (scalar_idx, scalar_bytes) in scalars.iter().enumerate() {
                let sub_table = &bases[scalar_idx * (1 << (get_window_size() - 1))
                    ..(scalar_idx + 1) * (1 << (get_window_size() - 1))];

                let point_idx =
                    get_booth_index(window_idx, get_window_size(), scalar_bytes.as_u8());

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
            for _ in 0..get_window_size() {
                result = result.dbl();
            }
            // Add the accumulated point for this window
            result.add_or_dbl_assign(&point);
        }

        result
    }

    pub fn multiply_sequential(&self, scalars: &[TFr]) -> TG1 {
        Self::multiply_sequential_raw(&self.points, scalars)
    }

    pub fn multiply_batch(&self, scalars: &[Vec<TFr>]) -> Vec<TG1> {
        assert!(scalars.len() == self.batch_points.len());

        #[cfg(not(feature = "parallel"))]
        {
            self.batch_points
                .iter()
                .zip(scalars)
                .map(|(points, scalars)| Self::multiply_sequential_raw(points, scalars))
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
                        Self::multiply_sequential_raw(&self.batch_points[work], &scalars[work]);
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
