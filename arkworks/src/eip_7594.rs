extern crate alloc;

use crate::consts::{CELL_INDICES_RBL, FIELD_ELEMENTS_PER_CELL, FIELD_ELEMENTS_PER_EXT_BLOB};
use crate::fft::fft_fr_fast;
use crate::kzg_proofs::{g1_linear_combination, pairings_verify};
use crate::kzg_proofs::{LFFTSettings, LKZGSettings};
use crate::kzg_types::{ArkFp, ArkFr, ArkG1, ArkG1Affine, ArkG2};
use crate::utils::PolyData;
use blst::blst_fr_mul;
use kzg::common_utils::{reverse_bit_order, reverse_bits_limited};
use kzg::eip_4844::{
    blob_to_kzg_commitment_rust, blob_to_polynomial, compute_blob_kzg_proof_rust,
    compute_kzg_proof_rust, compute_powers, hash, hash_to_bls_field, load_trusted_setup_rust,
    verify_blob_kzg_proof_batch_rust, verify_blob_kzg_proof_rust, verify_kzg_proof_rust, Blob,
    Bytes32, Bytes48, CKZGSettings, Cell, KZGCommitment, KZGProof, PrecomputationTableManager,
    BYTES_PER_CELL, BYTES_PER_COMMITMENT, BYTES_PER_FIELD_ELEMENT, BYTES_PER_G1, BYTES_PER_G2,
    BYTES_PER_PROOF, CELLS_PER_EXT_BLOB, C_KZG_RET, C_KZG_RET_BADARGS, C_KZG_RET_OK,
    FIELD_ELEMENTS_PER_BLOB, RANDOM_CHALLENGE_KZG_CELL_BATCH_DOMAIN, TRUSTED_SETUP_NUM_G1_POINTS,
    TRUSTED_SETUP_NUM_G2_POINTS,
};
use kzg::{cfg_into_iter, Fr, G1Mul, KZGSettings, G1, G2};
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

use crate::fft_g1::fft_g1_fast;
use crate::utils::kzg_settings_to_rust;

fn fr_ifft(output: &mut [ArkFr], input: &[ArkFr], s: &LFFTSettings) -> Result<(), String> {
    let stride = kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB / input.len();

    fft_fr_fast(output, input, 1, &s.reverse_roots_of_unity, stride);

    let inv_len = ArkFr::from_u64(input.len().try_into().unwrap()).inverse();
    for el in output {
        *el = el.mul(&inv_len);
    }

    Ok(())
}

fn fr_fft(output: &mut [ArkFr], input: &[ArkFr], s: &LFFTSettings) -> Result<(), String> {
    let roots_stride = kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB / input.len();
    fft_fr_fast(output, input, 1, &s.roots_of_unity, roots_stride);

    Ok(())
}

fn g1_ifft(out: &mut [ArkG1], input: &[ArkG1], s: &LKZGSettings) -> Result<(), String> {
    if input.len() > kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB || !input.len().is_power_of_two() {
        return Err("Invalid input length".to_string());
    }

    let stride = kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB / input.len();
    fft_g1_fast(out, input, 1, &s.fs.reverse_roots_of_unity, stride);

    let inv_len = ArkFr::from_u64(input.len() as u64).eucl_inverse();
    for out in out.iter_mut().take(input.len()) {
        *out = out.mul(&inv_len);
    }

    Ok(())
}

fn g1_fft(out: &mut [ArkG1], input: &[ArkG1], s: &LKZGSettings) -> Result<(), String> {
    if input.len() > kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB || !input.len().is_power_of_two() {
        return Err("Invalid input length".to_string());
    }

    let roots_stride = kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB / input.len();
    fft_g1_fast(out, input, 1, &s.fs.roots_of_unity, roots_stride);

    Ok(())
}

fn toeplitz_coeffs_stride(
    out: &mut [ArkFr],
    input: &[ArkFr],
    n: usize,
    offset: usize,
    stride: usize,
) -> Result<(), String> {
    if stride == 0 {
        return Err("Stride cannot be zero".to_string());
    }
    // fr_t *out, const fr_t *in, size_t n, size_t offset, size_t stride

    // size_t k, k2;

    // if (stride == 0) return C_KZG_BADARGS;

    let k = n / stride;
    let k2 = k * 2;
    // k = n / stride;
    // k2 = k * 2;

    out[0] = input[n - 1 - offset];
    // out[0] = in[n - 1 - offset];
    // for i in 1..
    {
        let mut i = 1;
        while i <= k + 1 && i < k2 {
            out[i] = ArkFr::zero();
            i += 1;
        }
    };

    {
        let mut i = k + 2;
        let mut j = 2 * stride - offset - 1;
        while i < k2 {
            out[i] = input[j];
            i += 1;
            j += stride;
        }
    };
    // for (size_t i = 1; i <= k + 1 && i < k2; i++) {
    //     out[i] = FR_ZERO;
    // }
    // for (size_t i = k + 2, j = 2 * stride - offset - 1; i < k2; i++, j += stride) {
    //     out[i] = in[j];
    // }

    // return C_KZG_OK;
    Ok(())
}

fn get_coset_shift_pow_for_cell(
    cell_index: usize,
    settings: &LKZGSettings,
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

fn poly_lagrange_to_monomial(
    output: &mut [ArkFr],
    largrange_poly: &[ArkFr],
    s: &LFFTSettings,
) -> Result<(), String> {
    let mut poly = largrange_poly.to_vec();

    reverse_bit_order(&mut poly)?;

    fr_ifft(output, &poly, s)?;

    Ok(())
}

fn coset_fft(input: &[ArkFr], s: &LKZGSettings) -> Result<Vec<ArkFr>, String> {
    if input.is_empty() {
        return Err("Invalid input length".to_string());
    }

    let mut in_shifted = input.to_vec();
    // TODO: move 7 to constant
    shift_poly(&mut in_shifted, &ArkFr::from_u64(7));

    let mut output = vec![ArkFr::default(); input.len()];
    fr_fft(&mut output, &in_shifted, &s.fs)?;

    Ok(output)
}

fn coset_ifft(input: &[ArkFr], s: &LKZGSettings) -> Result<Vec<ArkFr>, String> {
    if input.is_empty() {
        return Err("Invalid input length".to_string());
    }

    let mut output = vec![ArkFr::default(); input.len()];
    fr_ifft(&mut output, input, &s.fs)?;

    shift_poly(&mut output, &ArkFr::one().div(&ArkFr::from_u64(7))?);

    Ok(output)
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

fn compute_vanishing_polynomial_from_roots(roots: &[ArkFr]) -> Result<Vec<ArkFr>, String> {
    if roots.is_empty() {
        return Err("Roots cannot be empty".to_string());
    }

    let mut poly = Vec::new();
    poly.push(roots[0].negate());

    for i in 1..roots.len() {
        let neg_root = roots[i].negate();

        poly.push(neg_root.add(&poly[i - 1]));

        for j in (1..i).rev() {
            poly[j] = poly[j].mul(&neg_root).add(&poly[j - 1]);
        }
        poly[0] = poly[0].mul(&neg_root);
    }

    poly.push(ArkFr::one());

    Ok(poly)
}

fn vanishing_polynomial_for_missing_cells(
    missing_cell_indicies: &[usize],
    s: &LKZGSettings,
) -> Result<Vec<ArkFr>, String> {
    if missing_cell_indicies.is_empty() || missing_cell_indicies.len() >= CELLS_PER_EXT_BLOB {
        return Err("Invalid missing cell indicies count".to_string());
    }

    const STRIDE: usize = kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB / CELLS_PER_EXT_BLOB;

    let roots = missing_cell_indicies
        .iter()
        .map(|i| s.get_roots_of_unity_at(*i * STRIDE))
        .collect::<Vec<_>>();

    let short_vanishing_poly = compute_vanishing_polynomial_from_roots(&roots)?;

    let mut vanishing_poly = vec![ArkFr::zero(); kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB];

    for i in 0..short_vanishing_poly.len() {
        vanishing_poly[i * kzg::eip_4844::FIELD_ELEMENTS_PER_CELL] = short_vanishing_poly[i];
    }

    Ok(vanishing_poly)
}

fn shift_poly(poly: &mut [ArkFr], shift_factor: &ArkFr) {
    let mut factor_power = ArkFr::one();
    for coeff in poly.iter_mut().skip(1) {
        factor_power = factor_power.mul(shift_factor);
        *coeff = coeff.mul(&factor_power);
    }
}

fn recover_cells(
    output: &mut [ArkFr],
    cell_indicies: &[usize],
    s: &LKZGSettings,
) -> Result<(), String> {
    let mut missing_cell_indicies = Vec::new();

    let mut cells_brp = output.to_vec();
    reverse_bit_order(&mut cells_brp)?;

    for i in 0..CELLS_PER_EXT_BLOB {
        if !cell_indicies.contains(&i) {
            missing_cell_indicies.push(reverse_bits_limited(CELLS_PER_EXT_BLOB, i));
        }
    }

    let missing_cell_indicies = &missing_cell_indicies[..];

    if missing_cell_indicies.len() > CELLS_PER_EXT_BLOB / 2 {
        return Err("Not enough cells".to_string());
    }

    let vanishing_poly_coeff = vanishing_polynomial_for_missing_cells(missing_cell_indicies, s)?;
    let mut vanishing_poly_eval =
        vec![ArkFr::default(); kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB];

    fr_fft(&mut vanishing_poly_eval, &vanishing_poly_coeff, &s.fs)?;

    let mut extended_evaluation_times_zero =
        Vec::with_capacity(kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB);

    for i in 0..kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB {
        if cells_brp[i].is_null() {
            extended_evaluation_times_zero.push(ArkFr::zero());
        } else {
            extended_evaluation_times_zero.push(cells_brp[i].mul(&vanishing_poly_eval[i]));
        }
    }

    let mut extended_evaluation_times_zero_coeffs =
        vec![ArkFr::default(); kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB];
    fr_ifft(
        &mut extended_evaluation_times_zero_coeffs,
        &extended_evaluation_times_zero,
        &s.fs,
    )?;

    let mut extended_evaluations_over_coset = coset_fft(&extended_evaluation_times_zero_coeffs, s)?;

    let vanishing_poly_over_coset = coset_fft(&vanishing_poly_coeff, s)?;

    for i in 0..kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB {
        extended_evaluations_over_coset[i] =
            extended_evaluations_over_coset[i].div(&vanishing_poly_over_coset[i])?;
    }

    let reconstructed_poly_coeff = coset_ifft(&extended_evaluations_over_coset, s)?;

    fr_fft(output, &reconstructed_poly_coeff, &s.fs)?;

    reverse_bit_order(output)?;

    Ok(())
}

fn compute_weighted_sum_of_commitments(
    commitments: &[ArkG1],
    commitment_indices: &[usize],
    r_powers: &[ArkFr],
) -> ArkG1 {
    let mut commitment_weights = vec![ArkFr::zero(); commitments.len()];

    for i in 0..r_powers.len() {
        commitment_weights[commitment_indices[i]] =
            commitment_weights[commitment_indices[i]].add(&r_powers[i]);
    }

    let mut sum_of_commitments = ArkG1::default();
    g1_linear_combination(
        &mut sum_of_commitments,
        commitments,
        &commitment_weights,
        commitments.len(),
        None,
    );

    sum_of_commitments
}

fn compute_fk20_proofs(
    proofs: &mut [ArkG1],
    poly: &[ArkFr],
    n: usize,
    s: &LKZGSettings,
) -> Result<(), String> {
    let k = n / kzg::eip_4844::FIELD_ELEMENTS_PER_CELL;
    let k2 = k * 2;

    let mut coeffs = vec![vec![ArkFr::default(); k]; k2];
    let mut h_ext_fft = vec![ArkG1::identity(); k2];
    let mut toeplitz_coeffs = vec![ArkFr::default(); k2];
    let mut toeplitz_coeffs_fft = vec![ArkFr::default(); k2];

    for i in 0..kzg::eip_4844::FIELD_ELEMENTS_PER_CELL {
        toeplitz_coeffs_stride(
            &mut toeplitz_coeffs,
            poly,
            n,
            i,
            kzg::eip_4844::FIELD_ELEMENTS_PER_CELL,
        )?;
        fr_fft(&mut toeplitz_coeffs_fft, &toeplitz_coeffs, &s.fs)?;
        for j in 0..k2 {
            coeffs[j][i] = toeplitz_coeffs_fft[j];
        }
    }

    for i in 0..k2 {
        g1_linear_combination(
            &mut h_ext_fft[i],
            &s.x_ext_fft_columns[i],
            &coeffs[i],
            kzg::eip_4844::FIELD_ELEMENTS_PER_CELL,
            None,
        );
    }

    let mut h = vec![ArkG1::identity(); k2];
    g1_ifft(&mut h, &h_ext_fft, s)?;

    for h in h.iter_mut().take(k2).skip(k) {
        *h = ArkG1::identity();
    }

    g1_fft(proofs, &h, s)?;

    Ok(())
}

pub fn compute_cells_and_kzg_proofs_rust(
    cells: Option<&mut [[ArkFr; kzg::eip_4844::FIELD_ELEMENTS_PER_CELL]]>,
    proofs: Option<&mut [ArkG1]>,
    blob: &[ArkFr],
    s: &LKZGSettings,
) -> Result<(), String> {
    if cells.is_none() && proofs.is_none() {
        return Err("Both cells & proofs cannot be none".to_string());
    }

    let poly = blob_to_polynomial::<ArkFr, PolyData>(blob)?;

    let mut poly_monomial = vec![ArkFr::zero(); kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB];

    poly_lagrange_to_monomial(
        &mut poly_monomial[..FIELD_ELEMENTS_PER_BLOB],
        &poly.coeffs,
        &s.fs,
    )?;

    // compute cells
    if let Some(cells) = cells {
        fr_fft(cells.as_flattened_mut(), &poly_monomial, &s.fs)?;

        reverse_bit_order(cells.as_flattened_mut())?;
    };

    // compute proofs
    if let Some(proofs) = proofs {
        compute_fk20_proofs(proofs, &poly_monomial, FIELD_ELEMENTS_PER_BLOB, s)?;
        reverse_bit_order(proofs)?;
    }

    Ok(())
}

pub fn recover_cells_and_kzg_proofs_rust(
    recovered_cells: &mut [[ArkFr; FIELD_ELEMENTS_PER_CELL]],
    recovered_proofs: Option<&mut [ArkG1]>,
    cell_indicies: &[usize],
    cells: &[[ArkFr; FIELD_ELEMENTS_PER_CELL]],
    s: &LKZGSettings,
) -> Result<(), String> {
    kzg::eip_7594::recover_cells_and_kzg_proofs(
        recovered_cells,
        recovered_proofs,
        cell_indicies,
        cells,
        s,
    )
}

fn computed_weighted_sum_of_proofs(
    proofs: &[ArkG1],
    r_powers: &[ArkFr],
    cell_indices: &[usize],
    s: &LKZGSettings,
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

fn get_inv_coset_shift_for_cell(
    cell_index: usize,
    settings: &LKZGSettings,
) -> Result<ArkFr, String> {
    /*
     * Get the cell index in reverse-bit order.
     * This index points to this cell's coset factor h_k in the roots_of_unity array.
     */
    let cell_index_rbl = CELL_INDICES_RBL[cell_index];

    /*
     * Observe that for every element in roots_of_unity, we can find its inverse by
     * accessing its reflected element.
     *
     * For example, consider a multiplicative subgroup with eight elements:
     *   roots = {w^0, w^1, w^2, ... w^7, w^0}
     * For a root of unity in roots[i], we can find its inverse in roots[-i].
     */
    if cell_index_rbl > kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB {
        return Err("Invalid cell index".to_string());
    }
    let inv_coset_factor_idx = kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB - cell_index_rbl;

    /* Get h_k^{-1} using the index */
    if inv_coset_factor_idx > kzg::eip_4844::FIELD_ELEMENTS_PER_EXT_BLOB {
        return Err("Invalid cell index".to_string());
    }

    Ok(settings.get_roots_of_unity_at(inv_coset_factor_idx))
}

fn compute_commitment_to_aggregated_interpolation_poly(
    r_powers: &[ArkFr],
    cell_indices: &[usize],
    cells: &[[ArkFr; kzg::eip_4844::FIELD_ELEMENTS_PER_CELL]],
    s: &LKZGSettings,
) -> Result<ArkG1, String> {
    let mut aggregated_column_cells =
        vec![ArkFr::zero(); CELLS_PER_EXT_BLOB * kzg::eip_4844::FIELD_ELEMENTS_PER_CELL];

    for (cell_index, column_index) in cell_indices.iter().enumerate() {
        for fr_index in 0..kzg::eip_4844::FIELD_ELEMENTS_PER_CELL {
            let original_fr = cells[cell_index][fr_index];

            let scaled_fr = original_fr.mul(&r_powers[cell_index]);

            let array_index = column_index * kzg::eip_4844::FIELD_ELEMENTS_PER_CELL + fr_index;
            aggregated_column_cells[array_index] =
                aggregated_column_cells[array_index].add(&scaled_fr);
        }
    }

    let mut is_cell_used = [false; CELLS_PER_EXT_BLOB];

    for cell_index in cell_indices {
        is_cell_used[*cell_index] = true;
    }

    let mut aggregated_interpolation_poly =
        vec![ArkFr::zero(); kzg::eip_4844::FIELD_ELEMENTS_PER_CELL];
    let mut column_interpolation_poly =
        vec![ArkFr::default(); kzg::eip_4844::FIELD_ELEMENTS_PER_CELL];
    for (i, is_cell_used) in is_cell_used.iter().enumerate() {
        if !is_cell_used {
            continue;
        }

        let index = i * kzg::eip_4844::FIELD_ELEMENTS_PER_CELL;

        reverse_bit_order(
            &mut aggregated_column_cells[index..(index + kzg::eip_4844::FIELD_ELEMENTS_PER_CELL)],
        )?;

        fr_ifft(
            &mut column_interpolation_poly,
            &aggregated_column_cells[index..(index + kzg::eip_4844::FIELD_ELEMENTS_PER_CELL)],
            &s.fs,
        )?;

        let inv_coset_factor = get_inv_coset_shift_for_cell(i, s)?;

        shift_poly(&mut column_interpolation_poly, &inv_coset_factor);

        for k in 0..kzg::eip_4844::FIELD_ELEMENTS_PER_CELL {
            aggregated_interpolation_poly[k] =
                aggregated_interpolation_poly[k].add(&column_interpolation_poly[k]);
        }
    }

    let mut commitment_out = ArkG1::default();
    g1_linear_combination(
        &mut commitment_out,
        &s.g1_values_monomial,
        &aggregated_interpolation_poly,
        kzg::eip_4844::FIELD_ELEMENTS_PER_CELL,
        None,
    ); // TODO: maybe pass precomputation here?

    Ok(commitment_out)
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
    s: &LKZGSettings,
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
        compute_weighted_sum_of_commitments(unique_commitments, &commitment_indices, &r_powers);

    let interpolation_poly_commit =
        compute_commitment_to_aggregated_interpolation_poly(&r_powers, cell_indices, cells, s)?;

    let final_g1_sum = final_g1_sum.sub(&interpolation_poly_commit);

    let weighted_sum_of_proofs =
        computed_weighted_sum_of_proofs(proofs, &r_powers, cell_indices, s)?;

    let final_g1_sum = final_g1_sum.add(&weighted_sum_of_proofs);

    let power_of_s = s.g2_values_monomial[FIELD_ELEMENTS_PER_CELL];

    Ok(pairings_verify(
        &final_g1_sum,
        &ArkG2::generator(),
        &proof_lincomb,
        &power_of_s,
    ))
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn recover_cells_and_kzg_proofs(
    recovered_cells: *mut Cell,
    recovered_proofs: *mut KZGProof,
    cell_indices: *const u64,
    cells: *const Cell,
    num_cells: u64,
    s: *const CKZGSettings,
) -> C_KZG_RET {
    unsafe fn inner(
        recovered_cells: *mut Cell,
        recovered_proofs: *mut KZGProof,
        cell_indices: *const u64,
        cells: *const Cell,
        num_cells: u64,
        s: *const CKZGSettings,
    ) -> Result<(), String> {
        let mut recovered_cells_rs =
            vec![[ArkFr::default(); FIELD_ELEMENTS_PER_CELL]; CELLS_PER_EXT_BLOB];

        let mut recovered_proofs_rs = if recovered_proofs.is_null() {
            None
        } else {
            Some(vec![ArkG1::default(); CELLS_PER_EXT_BLOB])
        };

        let cell_indicies = core::slice::from_raw_parts(cell_indices, num_cells as usize)
            .iter()
            .map(|it| *it as usize)
            .collect::<Vec<_>>();
        let cells = core::slice::from_raw_parts(cells, num_cells as usize)
            .iter()
            .map(|it| -> Result<[ArkFr; FIELD_ELEMENTS_PER_CELL], String> {
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
        let settings = kzg_settings_to_rust(&*s)?;

        recover_cells_and_kzg_proofs_rust(
            &mut recovered_cells_rs,
            recovered_proofs_rs.as_deref_mut(),
            &cell_indicies,
            &cells,
            &settings,
        )?;

        let recovered_cells = core::slice::from_raw_parts_mut(recovered_cells, CELLS_PER_EXT_BLOB);
        for (cell_c, cell_rs) in recovered_cells.iter_mut().zip(recovered_cells_rs.iter()) {
            cell_c.bytes.copy_from_slice(
                &cell_rs
                    .iter()
                    .flat_map(|fr| fr.to_bytes())
                    .collect::<Vec<_>>(),
            );
        }

        if let Some(recovered_proofs_rs) = recovered_proofs_rs {
            let recovered_proofs =
                core::slice::from_raw_parts_mut(recovered_proofs, CELLS_PER_EXT_BLOB);

            for (proof_c, proof_rs) in recovered_proofs.iter_mut().zip(recovered_proofs_rs.iter()) {
                proof_c.bytes = proof_rs.to_bytes();
            }
        }

        Ok(())
    }

    match inner(
        recovered_cells,
        recovered_proofs,
        cell_indices,
        cells,
        num_cells,
        s,
    ) {
        Ok(()) => C_KZG_RET_OK,
        Err(_) => C_KZG_RET_BADARGS,
    }
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
        let commitments = core::slice::from_raw_parts(commitments_bytes, num_cells as usize)
            .iter()
            .map(|bytes| ArkG1::from_bytes(&bytes.bytes))
            .collect::<Result<Vec<_>, String>>()?;

        let cell_indices = core::slice::from_raw_parts(cell_indices, num_cells as usize)
            .iter()
            .map(|it| *it as usize)
            .collect::<Vec<_>>();

        let cells = core::slice::from_raw_parts(cells, num_cells as usize)
            .iter()
            .map(
                |it| -> Result<[ArkFr; kzg::eip_4844::FIELD_ELEMENTS_PER_CELL], String> {
                    it.bytes
                        .chunks(BYTES_PER_FIELD_ELEMENT)
                        .map(ArkFr::from_bytes)
                        .collect::<Result<Vec<_>, String>>()
                        .and_then(|frs| {
                            frs.try_into()
                                .map_err(|_| "Invalid field element count per cell".to_string())
                        })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let proofs = core::slice::from_raw_parts(proofs_bytes, num_cells as usize)
            .iter()
            .map(|bytes| ArkG1::from_bytes(&bytes.bytes))
            .collect::<Result<Vec<_>, String>>()?;

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
