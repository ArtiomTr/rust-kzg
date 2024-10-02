extern crate alloc;

use crate::consts::{
    FIELD_ELEMENTS_PER_EXT_BLOB, CELL_INDICES_RBL, FIELD_ELEMENTS_PER_CELL
};
use crate::kzg_proofs::{FFTSettings, KZGSettings, g1_linear_combination};
use crate::kzg_types::{
    ArkFp, ArkFr, ArkG1, ArkG1Affine, ArkG2
};
use blst::{blst_fr_mul};
use kzg::common_utils::reverse_bit_order;
use kzg::eip_4844::{
    blob_to_kzg_commitment_rust, compute_blob_kzg_proof_rust, compute_kzg_proof_rust,
    load_trusted_setup_rust, verify_blob_kzg_proof_batch_rust, verify_blob_kzg_proof_rust,
    verify_kzg_proof_rust, Blob, Bytes32, Bytes48, CKZGSettings, KZGCommitment, KZGProof,
    PrecomputationTableManager, BYTES_PER_FIELD_ELEMENT, BYTES_PER_G1, BYTES_PER_G2, C_KZG_RET,
    C_KZG_RET_BADARGS, C_KZG_RET_OK, FIELD_ELEMENTS_PER_BLOB, TRUSTED_SETUP_NUM_G1_POINTS,
    TRUSTED_SETUP_NUM_G2_POINTS, Cell, CELLS_PER_EXT_BLOB, RANDOM_CHALLENGE_KZG_CELL_BATCH_DOMAIN,
    BYTES_PER_COMMITMENT, BYTES_PER_CELL, BYTES_PER_PROOF, hash, hash_to_bls_field, compute_powers
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

fn get_coset_shift_pow_for_cell(
    cell_index: usize,
    settings: &CKZGSettings,
) -> Result<ArkFr, String> {
    /*
     * Get the cell index in reverse-bit order.
     * This index points to this cell's coset factor h_k in the roots_of_unity array.
     */
    let cell_idx_rbl = CELL_INDICES_RBL[cell_index];

    /*
     * Get the index to h_k^n in the roots_of_unity array.
     *
     * Multiplying the index of h_k by n, effectively raises h_k to the n-th power,
     * because advancing in the roots_of_unity array corresponds to increasing exponents.
     */
    let h_k_pow_idx = cell_idx_rbl * kzg::eip_4844::FIELD_ELEMENTS_PER_CELL;

    if h_k_pow_idx > kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB {
        return Err("Invalid cell index".to_string());
    }

    /* Get h_k^n using the index */
    Ok(settings.get_roots_of_unity_at(h_k_pow_idx))
}

fn deduplicate_commitments(commitments: &mut [ArkG1], indicies: &mut [usize], count: &mut usize) {
    if *count == 0 {
        return;
    }

    indicies[0] = 0;
    let mut new_count = 1;

    for i in 1..*count {
        let mut exist = false;
        for j in 0..new_count {
            if commitments[i] == commitments[j] {
                indicies[i] = j;
                exist = true;
                break;
            }
        }

        if !exist {
            commitments[new_count] = commitments[i];
            indicies[i] = new_count;
            new_count += 1;
        }
    }
}

fn computed_weighted_sum_of_proofs(
    proofs: &[ArkG1],
    r_powers: &[ArkFr],
    cell_indices: &[usize],
    s: &CKZGSettings,
) -> Result<ArkG1, String> {
    let num_cells = proofs.len();

    if r_powers.len() != num_cells || cell_indices.len() != num_cells {
        return Err("Length mismatch".to_string());
    }

    let mut weighted_powers_of_r = Vec::with_capacity(num_cells);
    for i in 0..num_cells {
        let h_k_pow = get_coset_shift_pow_for_cell(cell_indices[i], s)?;

        weighted_powers_of_r.push(r_powers[i].mul(&h_k_pow));
    }

    let mut weighted_proofs_sum_out = ArkG1::default();
    g1_linear_combination(
        &mut weighted_proofs_sum_out,
        proofs,
        &weighted_powers_of_r,
        num_cells,
        None,
    );

    Ok(weighted_proofs_sum_out)
}

fn compute_r_powers_for_verify_cell_kzg_proof_batch(
    commitments: &[ArkG1],
    commitment_indices: &[usize],
    cell_indices: &[usize],
    cells: &[[ArkFr; kzg::eip_4844::FIELD_ELEMENTS_PER_CELL]],
    proofs: &[ArkG1],
) -> Result<Vec<ArkFr>, String> {
    if commitment_indices.len() != cells.len()
        || cell_indices.len() != cells.len()
        || proofs.len() != cells.len()
    {
        return Err("Cell count mismatch".to_string());
    }

    let input_size = RANDOM_CHALLENGE_KZG_CELL_BATCH_DOMAIN.len()
        + size_of::<u64>()
        + size_of::<u64>()
        + size_of::<u64>()
        + (commitments.len() * BYTES_PER_COMMITMENT)
        + (cells.len() * size_of::<u64>())
        + (cells.len() * size_of::<u64>())
        + (cells.len() * BYTES_PER_CELL)
        + (cells.len() * BYTES_PER_PROOF);

    let mut bytes = vec![0; input_size];
    bytes[..16].copy_from_slice(&RANDOM_CHALLENGE_KZG_CELL_BATCH_DOMAIN);
    bytes[16..24].copy_from_slice(&(kzg::eip_4844::FIELD_ELEMENTS_PER_CELL as u64).to_be_bytes());
    bytes[24..32].copy_from_slice(&(commitments.len() as u64).to_be_bytes());
    bytes[32..40].copy_from_slice(&(cells.len() as u64).to_be_bytes());

    let mut offset = 40;
    for commitment in commitments {
        bytes[offset..(offset + BYTES_PER_COMMITMENT)].copy_from_slice(&commitment.to_bytes());
        offset += BYTES_PER_COMMITMENT;
    }

    for i in 0..cells.len() {
        bytes[offset..(offset + 8)].copy_from_slice(&(commitment_indices[i] as u64).to_be_bytes());
        offset += 8;

        bytes[offset..(offset + 8)].copy_from_slice(&(cell_indices[i] as u64).to_be_bytes());
        offset += 8;

        bytes[offset..(offset + BYTES_PER_CELL)].copy_from_slice(
            &cells[i]
                .iter()
                .flat_map(|fr| fr.to_bytes())
                .collect::<Vec<_>>(),
        );
        offset += BYTES_PER_CELL;

        bytes[offset..(offset + BYTES_PER_PROOF)].copy_from_slice(&(proofs[i].to_bytes()));
        offset += BYTES_PER_PROOF;
    }

    let bytes = &bytes[..];

    if offset != input_size {
        return Err("Failed to create challenge - invalid length".to_string());
    }

    let eval_challenge = hash(bytes);
    let r = hash_to_bls_field(&eval_challenge);

    Ok(compute_powers(&r, cells.len()))
}

pub fn verify_cell_kzg_proof_batch_rust(
    commitments: &[ArkG1],
    cell_indices: &[usize],
    cells: &[[ArkFr; FIELD_ELEMENTS_PER_CELL]],
    proofs: &[ArkG1],
    s: &CKZGSettings,
) -> Result<bool, String> {
    if cells.len() != cell_indices.len() {
        return Err("Cell count mismatch".to_string());
    }

    if commitments.len() != cells.len() {
        return Err("Commitment count mismatch".to_string());
    }

    if proofs.len() != cells.len() {
        return Err("Proof count mismatch".to_string());
    }

    if cells.is_empty() {
        return Ok(true);
    }

    for cell_index in cell_indices {
        if *cell_index >= CELLS_PER_EXT_BLOB {
            return Err("Invalid cell index".to_string());
        }
    }

    for proof in proofs {
        if !proof.is_valid() {
            return Err("Proof is not valid".to_string());
        }
    }

    let mut new_count = commitments.len();
    let mut unique_commitments = commitments.to_vec();
    let mut commitment_indices = vec![0usize; cells.len()];
    deduplicate_commitments(
        &mut unique_commitments,
        &mut commitment_indices,
        &mut new_count,
    );

    for commitment in unique_commitments.iter() {
        if !commitment.is_valid() {
            return Err("Commitment is not valid".to_string());
        }
    }

    let unique_commitments = &unique_commitments[0..new_count];

    let r_powers = compute_r_powers_for_verify_cell_kzg_proof_batch(
        unique_commitments,
        &commitment_indices,
        cell_indices,
        cells,
        proofs,
    )?;

    let mut proof_lincomb = ArkG1::default();
    g1_linear_combination(&mut proof_lincomb, proofs, &r_powers, cells.len(), None);

    let final_g1_sum =
        // Error
        compute_weighted_sum_of_commitments(unique_commitments, &commitment_indices, &r_powers);

    let interpolation_poly_commit =
        // Error
        compute_commitment_to_aggregated_interpolation_poly(&r_powers, cell_indices, cells, s)?;

    let final_g1_sum = final_g1_sum.sub(&interpolation_poly_commit);

    let weighted_sum_of_proofs =
        computed_weighted_sum_of_proofs(proofs, &r_powers, cell_indices, s)?;

    let final_g1_sum = final_g1_sum.add(&weighted_sum_of_proofs);

    let power_of_s = s.g2_values_monomial[FIELD_ELEMENTS_PER_CELL];

    // Error
    Ok(pairings_verify(
        &final_g1_sum,
        // Error
        &ArkG2::generator(),
        &proof_lincomb,
        &power_of_s,
    ))
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn verify_cell_kzg_proof_batch(
    ok: *mut bool,
    commitments_bytes: *const Bytes48,
    cell_indices: *const u64,
    cells: *const Cell,
    proofs_bytes: *const Bytes48,
    num_cells: u64,
    s: *const CKZGSettings,
) -> C_KZG_RET {
    unsafe fn inner(
        ok: *mut bool,
        commitments_bytes: *const Bytes48,
        cell_indices: *const u64,
        cells: *const Cell,
        proofs_bytes: *const Bytes48,
        num_cells: u64,
        s: *const CKZGSettings,
    ) -> Result<(), String> {
        let commitments =
            core::slice::from_raw_parts(commitments_bytes, num_cells as usize)
            .iter()
            .map(|bytes| ArkG1::from_bytes(&bytes.bytes))
            .collect::<Result<Vec<_>, String>>()?;

        let cell_indices = core::slice::from_raw_parts(cell_indices, num_cells as usize)
            .iter()
            .map(|it| *it as usize)
            .collect::<Vec<_>>();

        let cells = core::slice::from_raw_parts(cells, num_cells as usize)
            .iter()
            .map(|it| -> Result<[ArkFr; kzg::eip_4844::FIELD_ELEMENTS_PER_CELL], String> {
                it.bytes
                    .chunks(BYTES_PER_FIELD_ELEMENT)
                    .map(ArkFr::from_bytes)
                    .collect::<Result<Vec<_>, String>>()
                    .and_then(|frs| {
                        frs.try_into()
                            .map_err(|_| "Invalid field element count per cell".to_string())
                    })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let proofs = core::slice::from_raw_parts(proofs_bytes, num_cells as usize)
            .iter()
            .map(|bytes| ArkG1::from_bytes(&bytes.bytes))
            .collect::<Result<Vec<_>, String>>()?;

        // Error
        let settings = kzg_settings_to_rust(&*s)?;

        *ok = verify_cell_kzg_proof_batch_rust(
            &commitments,
            &cell_indices,
            &cells,
            &proofs,
            &settings,
        )?;

        Ok(())
    }

    match inner(
        ok,
        commitments_bytes,
        cell_indices,
        cells,
        proofs_bytes,
        num_cells,
        s,
    ) {
        Ok(()) => C_KZG_RET_OK,
        Err(_) => C_KZG_RET_BADARGS,
    }
}