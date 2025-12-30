use alloc::string::String;

use crate::{Fr, G1, G1Affine, G1Fp};

// ============================================================================
// NEW MSM TRAITS
// ============================================================================

/// Fixed-base MSM with precomputed tables.
///
/// The struct implementing this trait owns the precomputation table for a fixed set of bases.
/// Use this for repeated MSMs with the same base points (e.g., trusted setup points).
pub trait FixedBaseMSM<TFr: Fr, TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>:
    Sync + Send + Sized
{
    /// Configuration type for this MSM implementation
    type Config: Default + Clone;

    /// Create precomputation table from base points
    fn new(config: Self::Config, bases: &[TG1]) -> Result<Self, String>;

    /// MSM: sum(scalars[i] * bases[i])
    ///
    /// Uses parallel implementation if `parallel` feature is enabled and size is large enough.
    fn multiply(&self, scalars: &[TFr]) -> TG1;

    /// Sequential MSM, useful for outer parallelization.
    ///
    /// When running many small MSMs in parallel (e.g., batch of 128),
    /// use this method for each individual MSM to avoid nested parallelism.
    fn multiply_sequential(&self, scalars: &[TFr]) -> TG1;
}

/// Variable-base MSM without precomputation.
///
/// Points are provided at multiply time. Use this when the base points vary between calls.
pub trait VariableBaseMSM<TFr: Fr, TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>:
    Sync + Send + Default
{
    /// MSM with dynamically provided points: sum(scalars[i] * points[i])
    ///
    /// Uses parallel implementation if `parallel` feature is enabled and size is large enough.
    fn multiply(&self, points: &[TG1], scalars: &[TFr]) -> TG1;

    /// Sequential variant, useful for outer parallelization.
    fn multiply_sequential(&self, points: &[TG1], scalars: &[TFr]) -> TG1;
}

// ============================================================================
// NEW MODULES
// ============================================================================

pub mod batch;
pub mod pippenger;
pub mod utils;

// ============================================================================
// EXISTING MODULES (kept for transition)
// ============================================================================

pub mod arkmsm;
pub mod cell;
pub mod msm_impls;
pub mod precompute;
#[cfg(feature = "parallel")]
pub mod thread_pool;
pub mod types;

#[cfg(feature = "bgmw")]
pub mod bgmw;

#[cfg(feature = "sppark")]
mod sppark;

#[cfg(feature = "wbits")]
pub mod wbits;

#[cfg(all(feature = "diskcache", feature = "wbits"))]
mod diskcache;

// ============================================================================
// RE-EXPORTS
// ============================================================================

// Re-export key types for convenience
pub use batch::BatchFixedBaseMSM;
pub use pippenger::PippengerMSM;

// Re-export MSM functions for backward compatibility
pub use msm_impls::{msm, msm_batch};

#[cfg(feature = "bgmw")]
pub use bgmw::{BgmwConfig, BgmwTable};

#[cfg(feature = "wbits")]
pub use wbits::{WbitsConfig, WbitsTable};
