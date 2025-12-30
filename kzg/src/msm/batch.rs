//! Batch Fixed-Base MSM
//!
//! This module provides `BatchFixedBaseMSM`, a collection of `FixedBaseMSM` tables
//! for efficient batch operations. Each table handles one row of bases, and batch
//! multiplication parallelizes across rows.

use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{Fr, G1Affine, G1Fp, G1};

use super::FixedBaseMSM;

#[cfg(feature = "parallel")]
use super::{
    cell::Cell,
    thread_pool::{da_pool, ThreadPoolExt},
};

#[cfg(feature = "parallel")]
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "parallel")]
use alloc::sync::Arc;

// ============================================================================
// BATCH FIXED-BASE MSM
// ============================================================================

/// Collection of FixedBaseMSM tables for batch operations.
///
/// Each table handles one row of bases (e.g., 128 rows with ~4096 points each).
/// Batch multiplication parallelizes across rows using `multiply_sequential`
/// on each individual table to avoid nested parallelism.
///
/// # Type Parameters
/// - `TFr`: Field element type
/// - `TG1`: Group element type (projective)
/// - `TG1Fp`: Field element for G1 coordinates
/// - `TG1Affine`: Affine group element type
/// - `TTable`: The FixedBaseMSM implementation to use
///
/// # Example
/// ```ignore
/// // Create 128 tables using BGMW
/// let config = BgmwConfig::default();
/// let batch_msm = BatchFixedBaseMSM::<_, _, _, _, BgmwTable<_>>::new(config, &matrix)?;
///
/// // Multiply all 128 rows in parallel
/// let results = batch_msm.multiply_batch(&scalars_matrix);
/// ```
#[derive(Debug, Clone)]
pub struct BatchFixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine, TTable>
where
    TFr: Fr,
    TG1: G1,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TTable: FixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine>,
{
    tables: Vec<TTable>,
    _phantom: PhantomData<fn() -> (TFr, TG1, TG1Fp, TG1Affine)>,
}

impl<TFr, TG1, TG1Fp, TG1Affine, TTable> BatchFixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine, TTable>
where
    TFr: Fr,
    TG1: G1,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TTable: FixedBaseMSM<TFr, TG1, TG1Fp, TG1Affine>,
{
    /// Create a batch MSM from a matrix of points.
    ///
    /// Each row in the matrix becomes one FixedBaseMSM table.
    ///
    /// # Arguments
    /// - `config`: Configuration for the underlying MSM implementation
    /// - `matrix`: Matrix of points, one row per table
    ///
    /// # Returns
    /// A BatchFixedBaseMSM with one table per row.
    pub fn new(config: TTable::Config, matrix: &[Vec<TG1>]) -> Result<Self, String> {
        let mut tables = Vec::with_capacity(matrix.len());

        for row in matrix {
            let table = TTable::new(config.clone(), row)?;
            tables.push(table);
        }

        Ok(Self {
            tables,
            _phantom: PhantomData,
        })
    }

    /// Get the number of tables (rows).
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// Get a reference to a specific table.
    pub fn get_table(&self, index: usize) -> Option<&TTable> {
        self.tables.get(index)
    }

    /// Batch multiply: compute MSM for each row.
    ///
    /// When `parallel` feature is enabled, distributes work across thread pool.
    /// Each row uses `multiply_sequential` to avoid nested parallelism.
    ///
    /// # Arguments
    /// - `scalars`: Matrix of scalars, one row per table
    ///
    /// # Panics
    /// Panics if `scalars.len() != self.tables.len()`
    pub fn multiply_batch(&self, scalars: &[Vec<TFr>]) -> Vec<TG1> {
        assert_eq!(
            self.tables.len(),
            scalars.len(),
            "Batch size mismatch: expected {} rows, got {}",
            self.tables.len(),
            scalars.len()
        );

        #[cfg(feature = "parallel")]
        {
            self.multiply_batch_parallel(scalars)
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.multiply_batch_sequential(scalars)
        }
    }

    /// Sequential batch multiplication.
    fn multiply_batch_sequential(&self, scalars: &[Vec<TFr>]) -> Vec<TG1> {
        self.tables
            .iter()
            .zip(scalars.iter())
            .map(|(table, s)| table.multiply_sequential(s))
            .collect()
    }

    /// Parallel batch multiplication using thread pool.
    #[cfg(feature = "parallel")]
    fn multiply_batch_parallel(&self, scalars: &[Vec<TFr>]) -> Vec<TG1> {
        let pool = da_pool();
        let ncpus = pool.max_count();
        let total = scalars.len();

        // Fall back to sequential if not enough work
        if ncpus <= 1 || total < 2 {
            return self.multiply_batch_sequential(scalars);
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let mut results: Vec<Cell<TG1>> = Vec::with_capacity(total);
        #[allow(clippy::uninit_vec)]
        unsafe {
            results.set_len(results.capacity());
        }

        let results = &results[..];
        let n_workers = core::cmp::min(ncpus, total);

        for _ in 0..n_workers {
            let counter = counter.clone();

            pool.joined_execute(move || {
                loop {
                    let work = counter.fetch_add(1, Ordering::Relaxed);
                    if work >= total {
                        break;
                    }

                    let result = self.tables[work].multiply_sequential(&scalars[work]);
                    unsafe {
                        *results[work].as_ptr().as_mut().unwrap() = result;
                    }
                }
            });
        }

        pool.join();

        results.iter().map(|c| c.as_mut().clone()).collect()
    }
}
