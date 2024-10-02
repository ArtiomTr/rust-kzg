extern crate alloc;

use crate::consts::{
    FIELD_ELEMENTS_PER_EXT_BLOB, CELL_INDICES_RBL, FIELD_ELEMENTS_PER_CELL
};
use crate::kzg_proofs::{FFTSettings, KZGSettings};
use crate::kzg_types::{ArkFp, ArkFr as BlstFr, ArkG1, ArkG1Affine, ArkG2};
use blst::{blst_fr_mul};
use kzg::common_utils::reverse_bit_order;
use kzg::eip_4844::{
    blob_to_kzg_commitment_rust, compute_blob_kzg_proof_rust, compute_kzg_proof_rust,
    load_trusted_setup_rust, verify_blob_kzg_proof_batch_rust, verify_blob_kzg_proof_rust,
    verify_kzg_proof_rust, Blob, Bytes32, Bytes48, CKZGSettings, KZGCommitment, KZGProof,
    PrecomputationTableManager, BYTES_PER_FIELD_ELEMENT, BYTES_PER_G1, BYTES_PER_G2, C_KZG_RET,
    C_KZG_RET_BADARGS, C_KZG_RET_OK, FIELD_ELEMENTS_PER_BLOB, TRUSTED_SETUP_NUM_G1_POINTS,
    TRUSTED_SETUP_NUM_G2_POINTS
};
use kzg::{cfg_into_iter, Fr, G1};
use std::ptr::null_mut;

#[cfg(feature = "std")]
use libc::FILE;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Read;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "std")]
use kzg::eip_4844::load_trusted_setup_string;

pub fn get_coset_shift_pow_for_cell(
    coset_factor_out: &mut [BlstFr],
    cell_index: usize,
    s: &CKZGSettings,
) {
    let cell_idx_rbl : usize = CELL_INDICES_RBL[cell_index];
    let h_k_pow_idx : usize = cell_idx_rbl * FIELD_ELEMENTS_PER_CELL;

    assert!(h_k_pow_idx < FIELD_ELEMENTS_PER_EXT_BLOB + 1);
    *coset_factor_out = (*s).roots_of_unity[h_k_pow_idx];
}

pub unsafe extern "C" fn computed_weighted_sum_of_proofs(
    weighted_proof_sum_out: &mut [ArkG1],
    proofs_g1: &[ArkG1],
    r_powers: &[BlstFr],
    cell_indices: *const usize,
    num_cells: usize,
    s: &CKZGSettings,
) -> C_KZG_RET {

    let weighted_powers_of_r : &mut [BlstFr] = NULL;
    let h_k_pow : &mut [BlstFr] = NULL;

    //ret = new_fr_array(&weighted_powers_of_r, num_cells);
    //if (ret != C_KZG_OK) goto out;

    for i in 0..num_cells {
        // Get scaling factor h_k^n where h_k is the coset factor for this cell *
        get_coset_shift_pow_for_cell(h_k_pow, cell_indices[i], s);

        // Scale the power of r by h_k^n */
        blst_fr_mul(&weighted_powers_of_r[i], &r_powers[i], &h_k_pow);
    }

    //ret = g1_lincomb_fast(weighted_proof_sum_out, proofs_g1, weighted_powers_of_r, num_cells);

    //out:
    //    c_kzg_free(weighted_powers_of_r);
    //return ret;

    C_KZG_RET_OK
}
