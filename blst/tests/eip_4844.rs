#[cfg(test)]
mod tests {

    use kzg::eip_4844::FIELD_ELEMENTS_PER_BLOB;
    use kzg::{Fr, Poly};

    use kzg_bench::tests::eip_4844::{
        blob_to_kzg_commitment_test, bytes_to_bls_field_test,
        compute_and_verify_blob_kzg_proof_fails_with_incorrect_proof_test,
        compute_and_verify_blob_kzg_proof_test,
        compute_and_verify_kzg_proof_fails_with_incorrect_proof_test,
        compute_and_verify_kzg_proof_round_trip_test, compute_kzg_proof_test, compute_powers_test,
        verify_kzg_proof_batch_fails_with_incorrect_proof_test, verify_kzg_proof_batch_test,
    };
    #[cfg(not(feature = "minimal-spec"))]
    use kzg_bench::tests::eip_4844::{
        compute_and_verify_kzg_proof_within_domain_test, test_vectors_blob_to_kzg_commitment,
        test_vectors_compute_blob_kzg_proof, test_vectors_compute_kzg_proof,
        test_vectors_verify_blob_kzg_proof, test_vectors_verify_blob_kzg_proof_batch,
        test_vectors_verify_kzg_proof,
    };
    use kzg_bench::tests::utils::get_trusted_setup_path;
    use rust_kzg_blst::consts::SCALE2_ROOT_OF_UNITY;
    use rust_kzg_blst::eip_4844::{
        blob_to_kzg_commitment_rust, blob_to_polynomial_rust, bytes_to_blob,
        compute_blob_kzg_proof_rust, compute_kzg_proof_rust, compute_powers,
        evaluate_polynomial_in_evaluation_form_rust, load_trusted_setup_filename_rust,
        verify_blob_kzg_proof_batch_rust, verify_blob_kzg_proof_rust, verify_kzg_proof_rust,
    };
    use rust_kzg_blst::types::fft_settings::expand_root_of_unity;
    use rust_kzg_blst::types::{
        fft_settings::FsFFTSettings, fr::FsFr, g1::FsG1, g2::FsG2, kzg_settings::FsKZGSettings,
        poly::FsPoly,
    };

    #[test]
    pub fn bytes_to_bls_field_test_() {
        bytes_to_bls_field_test::<FsFr>();
    }

    #[test]
    pub fn compute_powers_test_() {
        compute_powers_test::<FsFr>(&compute_powers);
    }

    #[test]
    pub fn blob_to_kzg_commitment_test_() {
        blob_to_kzg_commitment_test::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
        );
    }

    #[test]
    pub fn compute_kzg_proof_test_() {
        compute_kzg_proof_test::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &compute_kzg_proof_rust,
            &blob_to_polynomial_rust,
            &evaluate_polynomial_in_evaluation_form_rust,
        );
    }

    #[test]
    pub fn compute_and_verify_kzg_proof_round_trip_test_() {
        compute_and_verify_kzg_proof_round_trip_test::<
            FsFr,
            FsG1,
            FsG2,
            FsPoly,
            FsFFTSettings,
            FsKZGSettings,
        >(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
            &compute_kzg_proof_rust,
            &blob_to_polynomial_rust,
            &evaluate_polynomial_in_evaluation_form_rust,
            &verify_kzg_proof_rust,
        );
    }

    #[cfg(not(feature = "minimal-spec"))]
    #[test]
    pub fn compute_and_verify_kzg_proof_within_domain_test_() {
        compute_and_verify_kzg_proof_within_domain_test::<
            FsFr,
            FsG1,
            FsG2,
            FsPoly,
            FsFFTSettings,
            FsKZGSettings,
        >(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
            &compute_kzg_proof_rust,
            &blob_to_polynomial_rust,
            &evaluate_polynomial_in_evaluation_form_rust,
            &verify_kzg_proof_rust,
        );
    }

    #[test]
    pub fn compute_and_verify_kzg_proof_fails_with_incorrect_proof_test_() {
        compute_and_verify_kzg_proof_fails_with_incorrect_proof_test::<
            FsFr,
            FsG1,
            FsG2,
            FsPoly,
            FsFFTSettings,
            FsKZGSettings,
        >(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
            &compute_kzg_proof_rust,
            &blob_to_polynomial_rust,
            &evaluate_polynomial_in_evaluation_form_rust,
            &verify_kzg_proof_rust,
        );
    }

    #[test]
    pub fn compute_and_verify_blob_kzg_proof_test_() {
        compute_and_verify_blob_kzg_proof_test::<
            FsFr,
            FsG1,
            FsG2,
            FsPoly,
            FsFFTSettings,
            FsKZGSettings,
        >(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
            &compute_blob_kzg_proof_rust,
            &verify_blob_kzg_proof_rust,
        );
    }

    #[test]
    pub fn compute_and_verify_blob_kzg_proof_fails_with_incorrect_proof_test_() {
        compute_and_verify_blob_kzg_proof_fails_with_incorrect_proof_test::<
            FsFr,
            FsG1,
            FsG2,
            FsPoly,
            FsFFTSettings,
            FsKZGSettings,
        >(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
            &compute_blob_kzg_proof_rust,
            &verify_blob_kzg_proof_rust,
        );
    }

    #[test]
    pub fn verify_kzg_proof_batch_test_() {
        verify_kzg_proof_batch_test::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
            &compute_blob_kzg_proof_rust,
            &verify_blob_kzg_proof_batch_rust,
        );
    }

    #[test]
    pub fn verify_kzg_proof_batch_fails_with_incorrect_proof_test_() {
        verify_kzg_proof_batch_fails_with_incorrect_proof_test::<
            FsFr,
            FsG1,
            FsG2,
            FsPoly,
            FsFFTSettings,
            FsKZGSettings,
        >(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
            &compute_blob_kzg_proof_rust,
            &verify_blob_kzg_proof_batch_rust,
        );
    }

    #[cfg(not(feature = "minimal-spec"))]
    #[test]
    pub fn test_vectors_blob_to_kzg_commitment_() {
        test_vectors_blob_to_kzg_commitment::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &blob_to_kzg_commitment_rust,
            &bytes_to_blob,
        );
    }

    #[cfg(not(feature = "minimal-spec"))]
    #[test]
    pub fn test_vectors_compute_kzg_proof_() {
        test_vectors_compute_kzg_proof::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &compute_kzg_proof_rust,
            &bytes_to_blob,
        );
    }

    #[cfg(not(feature = "minimal-spec"))]
    #[test]
    pub fn test_vectors_compute_blob_kzg_proof_() {
        test_vectors_compute_blob_kzg_proof::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &bytes_to_blob,
            &compute_blob_kzg_proof_rust,
        );
    }

    #[cfg(not(feature = "minimal-spec"))]
    #[test]
    pub fn test_vectors_verify_kzg_proof_() {
        test_vectors_verify_kzg_proof::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &verify_kzg_proof_rust,
        );
    }

    #[cfg(not(feature = "minimal-spec"))]
    #[test]
    pub fn test_vectors_verify_blob_kzg_proof_() {
        test_vectors_verify_blob_kzg_proof::<FsFr, FsG1, FsG2, FsPoly, FsFFTSettings, FsKZGSettings>(
            &load_trusted_setup_filename_rust,
            &bytes_to_blob,
            &verify_blob_kzg_proof_rust,
        );
    }

    #[cfg(not(feature = "minimal-spec"))]
    #[test]
    pub fn test_vectors_verify_blob_kzg_proof_batch_() {
        test_vectors_verify_blob_kzg_proof_batch::<
            FsFr,
            FsG1,
            FsG2,
            FsPoly,
            FsFFTSettings,
            FsKZGSettings,
        >(
            &load_trusted_setup_filename_rust,
            &bytes_to_blob,
            &verify_blob_kzg_proof_batch_rust,
        );
    }

    #[test]
    pub fn expand_root_of_unity_too_long() {
        let out = expand_root_of_unity(&FsFr::from_u64_arr(&SCALE2_ROOT_OF_UNITY[1]), 1);
        assert!(out.is_err());
    }

    #[test]
    pub fn expand_root_of_unity_too_short() {
        let out = expand_root_of_unity(&FsFr::from_u64_arr(&SCALE2_ROOT_OF_UNITY[1]), 3);
        assert!(out.is_err());
    }

    #[test]
    pub fn compute_kzg_proof_incorrect_blob_length() {
        let blob = &[FsFr::zero()];
        let out = blob_to_polynomial_rust(blob);
        // let = compute_kzg_proof_rust(blob, blob, load_kzg)
        assert!(out.is_err());
    }

    #[test]
    pub fn compute_kzg_proof_incorrect_poly_length() {
        let out = evaluate_polynomial_in_evaluation_form_rust(
            &Poly::new(1),
            &FsFr::zero(),
            &FsKZGSettings::default(),
        );

        assert!(out.is_err());
    }

    #[test]
    pub fn compute_kzg_proof_empty_blob_vector() {
        let res = verify_blob_kzg_proof_batch_rust(
            &[],
            &[FsG1::default()],
            &[FsG1::default()],
            &FsKZGSettings::default(),
        );

        assert!(res.is_ok());
    }

    #[test]
    pub fn compute_kzg_proof_incorrect_commitments_len() {
        let blob1 = vec![FsFr::default(); 3];
        let blob2 = vec![FsFr::default(); 3];

        let res = verify_blob_kzg_proof_batch_rust(
            &[blob1, blob2],
            &[FsG1::default()],
            &[FsG1::default(), FsG1::default()],
            &FsKZGSettings::default(),
        );

        assert!(res.is_err());
    }

    #[test]
    pub fn compute_kzg_proof_incorrect_proofs_len() {
        let blob1 = vec![FsFr::default(); 3];
        let blob2 = vec![FsFr::default(); 3];

        let res = verify_blob_kzg_proof_batch_rust(
            &[blob1, blob2],
            &[FsG1::default(), FsG1::default()],
            &[FsG1::default()],
            &FsKZGSettings::default(),
        );

        assert!(res.is_err());
    }

    #[test]
    pub fn validate_batched_input_test() {
        let path = get_trusted_setup_path();
        let setup = &load_trusted_setup_filename_rust(path.as_str()).unwrap();

        let blob1 = vec![FsFr::default(); FIELD_ELEMENTS_PER_BLOB];
        let blob2 = vec![FsFr::default(); FIELD_ELEMENTS_PER_BLOB];

        let res = verify_blob_kzg_proof_batch_rust(
            &[blob1, blob2],
            &[FsG1::default(), FsG1::default()],
            &[FsG1::default(), FsG1::default()],
            &setup,
        );

        assert!(res.is_ok());
    }
}
