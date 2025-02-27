extern crate alloc;

#[cfg(feature = "c_bindings")]
use alloc::{boxed::Box, vec::Vec};
#[cfg(feature = "c_bindings")]
use core::ptr;
use kzg::eip_4844::load_trusted_setup_rust;
#[cfg(feature = "c_bindings")]
use kzg::{
    eip_4844::{
        BYTES_PER_G1, FIELD_ELEMENTS_PER_BLOB, TRUSTED_SETUP_NUM_G1_POINTS,
        TRUSTED_SETUP_NUM_G2_POINTS,
    },
    eth::{
        c_bindings::{Blob, Bytes32, Bytes48, CKZGSettings, CKzgRet, KZGCommitment, KZGProof},
        FIELD_ELEMENTS_PER_CELL, FIELD_ELEMENTS_PER_EXT_BLOB,
    },
    Fr, G1,
};
#[cfg(all(feature = "std", feature = "c_bindings"))]
use libc::FILE;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Read;

#[cfg(feature = "std")]
use kzg::eip_4844::load_trusted_setup_string;

#[cfg(feature = "c_bindings")]
use crate::types::{fr::MclFr, g1::MclG1};

#[cfg(feature = "c_bindings")]
use crate::{types::kzg_settings::MclKZGSettings, utils::PRECOMPUTATION_TABLES};

#[cfg(feature = "c_bindings")]
macro_rules! handle_ckzg_badargs {
    ($x: expr) => {
        match $x {
            Ok(value) => value,
            Err(_) => return kzg::eth::c_bindings::CKzgRet::BadArgs,
        }
    };
}

#[cfg(feature = "c_bindings")]
fn kzg_settings_to_c(rust_settings: &MclKZGSettings) -> CKZGSettings {
    use kzg::eth::c_bindings::{blst_fp, blst_fp2, blst_fr, blst_p1, blst_p2};

    CKZGSettings {
        roots_of_unity: Box::leak(
            rust_settings
                .fs
                .roots_of_unity
                .iter()
                .map(|r| blst_fr {
                    l: r.to_blst_fr().l,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .as_mut_ptr(),
        brp_roots_of_unity: Box::leak(
            rust_settings
                .fs
                .brp_roots_of_unity
                .iter()
                .map(|r| blst_fr {
                    l: r.to_blst_fr().l,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .as_mut_ptr(),
        reverse_roots_of_unity: Box::leak(
            rust_settings
                .fs
                .reverse_roots_of_unity
                .iter()
                .map(|r| blst_fr {
                    l: r.to_blst_fr().l,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .as_mut_ptr(),
        g1_values_monomial: Box::leak(
            rust_settings
                .g1_values_monomial
                .iter()
                .map(|r| blst_p1 {
                    x: blst_fp { l: r.0.x.d },
                    y: blst_fp { l: r.0.y.d },
                    z: blst_fp { l: r.0.z.d },
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .as_mut_ptr(),
        g1_values_lagrange_brp: Box::leak(
            rust_settings
                .g1_values_lagrange_brp
                .iter()
                .map(|r| blst_p1 {
                    x: blst_fp { l: r.0.x.d },
                    y: blst_fp { l: r.0.y.d },
                    z: blst_fp { l: r.0.z.d },
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .as_mut_ptr(),
        g2_values_monomial: Box::leak(
            rust_settings
                .g2_values_monomial
                .iter()
                .map(|r| blst_p2 {
                    x: blst_fp2 {
                        fp: [blst_fp { l: r.0.x.d[0].d }, blst_fp { l: r.0.x.d[1].d }],
                    },
                    y: blst_fp2 {
                        fp: [blst_fp { l: r.0.y.d[0].d }, blst_fp { l: r.0.y.d[1].d }],
                    },
                    z: blst_fp2 {
                        fp: [blst_fp { l: r.0.z.d[0].d }, blst_fp { l: r.0.z.d[1].d }],
                    },
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .as_mut_ptr(),
        x_ext_fft_columns: Box::leak(
            rust_settings
                .x_ext_fft_columns
                .iter()
                .map(|r| {
                    Box::leak(
                        r.iter()
                            .map(|r| blst_p1 {
                                x: blst_fp { l: r.0.x.d },
                                y: blst_fp { l: r.0.y.d },
                                z: blst_fp { l: r.0.z.d },
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    )
                    .as_mut_ptr()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
        .as_mut_ptr(),
        tables: core::ptr::null_mut(),
        wbits: 0,
        scratch_size: 0,
    }
}

#[cfg(feature = "std")]
pub fn load_trusted_setup_filename_rust(
    filepath: &str,
) -> Result<crate::types::kzg_settings::MclKZGSettings, alloc::string::String> {
    let mut file = File::open(filepath).map_err(|_| "Unable to open file".to_string())?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|_| "Unable to read file".to_string())?;

    let (g1_monomial_bytes, g1_lagrange_bytes, g2_monomial_bytes) =
        load_trusted_setup_string(&contents)?;
    load_trusted_setup_rust(&g1_monomial_bytes, &g1_lagrange_bytes, &g2_monomial_bytes)
}

/// # Safety
#[cfg(feature = "c_bindings")]
#[no_mangle]
pub unsafe extern "C" fn blob_to_kzg_commitment(
    out: *mut KZGCommitment,
    blob: *const Blob,
    s: &CKZGSettings,
) -> CKzgRet {
    use kzg::eip_4844::blob_to_kzg_commitment_raw;

    let settings: MclKZGSettings = handle_ckzg_badargs!(s.try_into());
    let result = handle_ckzg_badargs!(blob_to_kzg_commitment_raw((*blob).bytes, &settings));
    (*out).bytes = result.to_bytes();

    CKzgRet::Ok
}

/// # Safety
#[cfg(feature = "c_bindings")]
#[no_mangle]
pub unsafe extern "C" fn load_trusted_setup(
    out: *mut CKZGSettings,
    g1_monomial_bytes: *const u8,
    num_g1_monomial_bytes: u64,
    g1_lagrange_bytes: *const u8,
    num_g1_lagrange_bytes: u64,
    g2_monomial_bytes: *const u8,
    num_g2_monomial_bytes: u64,
    _precompute: u64,
) -> CKzgRet {
    *out = CKZGSettings {
        brp_roots_of_unity: ptr::null_mut(),
        roots_of_unity: ptr::null_mut(),
        reverse_roots_of_unity: ptr::null_mut(),
        g1_values_monomial: ptr::null_mut(),
        g1_values_lagrange_brp: ptr::null_mut(),
        g2_values_monomial: ptr::null_mut(),
        x_ext_fft_columns: ptr::null_mut(),
        tables: ptr::null_mut(),
        wbits: 0,
        scratch_size: 0,
    };

    let g1_monomial_bytes =
        core::slice::from_raw_parts(g1_monomial_bytes, num_g1_monomial_bytes as usize);
    let g1_lagrange_bytes =
        core::slice::from_raw_parts(g1_lagrange_bytes, num_g1_lagrange_bytes as usize);
    let g2_monomial_bytes =
        core::slice::from_raw_parts(g2_monomial_bytes, num_g2_monomial_bytes as usize);
    TRUSTED_SETUP_NUM_G1_POINTS = num_g1_monomial_bytes as usize / BYTES_PER_G1;
    let mut settings = handle_ckzg_badargs!(load_trusted_setup_rust(
        g1_monomial_bytes,
        g1_lagrange_bytes,
        g2_monomial_bytes
    ));

    let c_settings = kzg_settings_to_c(&settings);

    PRECOMPUTATION_TABLES.save_precomputation(settings.precomputation.take(), &c_settings);

    *out = c_settings;
    CKzgRet::Ok
}

/// # Safety
#[cfg(all(feature = "std", feature = "c_bindings"))]
#[no_mangle]
pub unsafe extern "C" fn load_trusted_setup_file(
    out: *mut CKZGSettings,
    in_: *mut FILE,
) -> CKzgRet {
    *out = CKZGSettings {
        brp_roots_of_unity: ptr::null_mut(),
        roots_of_unity: ptr::null_mut(),
        reverse_roots_of_unity: ptr::null_mut(),
        g1_values_monomial: ptr::null_mut(),
        g1_values_lagrange_brp: ptr::null_mut(),
        g2_values_monomial: ptr::null_mut(),
        x_ext_fft_columns: ptr::null_mut(),
        tables: ptr::null_mut(),
        wbits: 0,
        scratch_size: 0,
    };

    let mut buf = vec![0u8; 1024 * 1024];
    let len: usize = libc::fread(buf.as_mut_ptr() as *mut libc::c_void, 1, buf.len(), in_);
    let s = handle_ckzg_badargs!(String::from_utf8(buf[..len].to_vec()));
    let (g1_monomial_bytes, g1_lagrange_bytes, g2_monomial_bytes) =
        handle_ckzg_badargs!(load_trusted_setup_string(&s));
    TRUSTED_SETUP_NUM_G1_POINTS = g1_monomial_bytes.len() / BYTES_PER_G1;
    if TRUSTED_SETUP_NUM_G1_POINTS != FIELD_ELEMENTS_PER_BLOB {
        // Helps pass the Java test "shouldThrowExceptionOnIncorrectTrustedSetupFromFile",
        // as well as 5 others that pass only if this one passes (likely because Java doesn't
        // deallocate its KZGSettings pointer when no exception is thrown).
        return CKzgRet::BadArgs;
    }
    let mut settings = handle_ckzg_badargs!(load_trusted_setup_rust(
        &g1_monomial_bytes,
        &g1_lagrange_bytes,
        &g2_monomial_bytes
    ));

    let c_settings = kzg_settings_to_c(&settings);

    PRECOMPUTATION_TABLES.save_precomputation(settings.precomputation.take(), &c_settings);

    *out = c_settings;

    CKzgRet::Ok
}

/// # Safety
#[no_mangle]
#[cfg(feature = "c_bindings")]
pub unsafe extern "C" fn compute_blob_kzg_proof(
    out: *mut KZGProof,
    blob: *const Blob,
    commitment_bytes: *const Bytes48,
    s: &CKZGSettings,
) -> CKzgRet {
    use kzg::eip_4844::compute_blob_kzg_proof_raw;

    let settings: MclKZGSettings = handle_ckzg_badargs!(s.try_into());
    let proof = handle_ckzg_badargs!(compute_blob_kzg_proof_raw(
        (*blob).bytes,
        (*commitment_bytes).bytes,
        &settings
    ));

    (*out).bytes = proof.to_bytes();
    CKzgRet::Ok
}

/// # Safety
#[no_mangle]
#[cfg(feature = "c_bindings")]
pub unsafe extern "C" fn free_trusted_setup(s: *mut CKZGSettings) {
    if s.is_null() {
        return;
    }

    PRECOMPUTATION_TABLES.remove_precomputation(&*s);

    if !(*s).g1_values_monomial.is_null() {
        let v = Box::from_raw(core::slice::from_raw_parts_mut(
            (*s).g1_values_monomial,
            FIELD_ELEMENTS_PER_BLOB,
        ));
        drop(v);
        (*s).g1_values_monomial = ptr::null_mut();
    }

    if !(*s).g1_values_lagrange_brp.is_null() {
        let v = Box::from_raw(core::slice::from_raw_parts_mut(
            (*s).g1_values_lagrange_brp,
            FIELD_ELEMENTS_PER_BLOB,
        ));
        drop(v);
        (*s).g1_values_lagrange_brp = ptr::null_mut();
    }

    if !(*s).g2_values_monomial.is_null() {
        let v = Box::from_raw(core::slice::from_raw_parts_mut(
            (*s).g2_values_monomial,
            TRUSTED_SETUP_NUM_G2_POINTS,
        ));
        drop(v);
        (*s).g2_values_monomial = ptr::null_mut();
    }

    if !(*s).x_ext_fft_columns.is_null() {
        let v = Box::from_raw(core::slice::from_raw_parts_mut(
            (*s).x_ext_fft_columns,
            2 * ((FIELD_ELEMENTS_PER_EXT_BLOB / 2) / FIELD_ELEMENTS_PER_CELL),
        ));
        drop(v);
        (*s).x_ext_fft_columns = ptr::null_mut();
    }

    if !(*s).roots_of_unity.is_null() {
        let v = Box::from_raw(core::slice::from_raw_parts_mut(
            (*s).roots_of_unity,
            FIELD_ELEMENTS_PER_EXT_BLOB + 1,
        ));
        drop(v);
        (*s).roots_of_unity = ptr::null_mut();
    }

    if !(*s).reverse_roots_of_unity.is_null() {
        let v = Box::from_raw(core::slice::from_raw_parts_mut(
            (*s).reverse_roots_of_unity,
            FIELD_ELEMENTS_PER_EXT_BLOB + 1,
        ));
        drop(v);
        (*s).reverse_roots_of_unity = ptr::null_mut();
    }

    if !(*s).brp_roots_of_unity.is_null() {
        let v = Box::from_raw(core::slice::from_raw_parts_mut(
            (*s).brp_roots_of_unity,
            FIELD_ELEMENTS_PER_EXT_BLOB,
        ));
        drop(v);
        (*s).brp_roots_of_unity = ptr::null_mut();
    }
}

/// # Safety
#[no_mangle]
#[cfg(feature = "c_bindings")]
pub unsafe extern "C" fn verify_kzg_proof(
    ok: *mut bool,
    commitment_bytes: *const Bytes48,
    z_bytes: *const Bytes32,
    y_bytes: *const Bytes32,
    proof_bytes: *const Bytes48,
    s: &CKZGSettings,
) -> CKzgRet {
    use kzg::eip_4844::verify_kzg_proof_rust;

    let frz = handle_ckzg_badargs!(MclFr::from_bytes(&(*z_bytes).bytes));
    let fry = handle_ckzg_badargs!(MclFr::from_bytes(&(*y_bytes).bytes));
    let g1commitment = handle_ckzg_badargs!(MclG1::from_bytes(&(*commitment_bytes).bytes));
    let g1proof = handle_ckzg_badargs!(MclG1::from_bytes(&(*proof_bytes).bytes));

    let settings: MclKZGSettings = handle_ckzg_badargs!(s.try_into());

    let result = handle_ckzg_badargs!(verify_kzg_proof_rust(
        &g1commitment,
        &frz,
        &fry,
        &g1proof,
        &settings
    ));

    *ok = result;
    CKzgRet::Ok
}

/// # Safety
#[no_mangle]
#[cfg(feature = "c_bindings")]
pub unsafe extern "C" fn verify_blob_kzg_proof(
    ok: *mut bool,
    blob: *const Blob,
    commitment_bytes: *const Bytes48,
    proof_bytes: *const Bytes48,
    s: &CKZGSettings,
) -> CKzgRet {
    use kzg::eip_4844::verify_blob_kzg_proof_raw;

    let settings: MclKZGSettings = handle_ckzg_badargs!(s.try_into());

    let result = handle_ckzg_badargs!(verify_blob_kzg_proof_raw(
        (*blob).bytes,
        (*commitment_bytes).bytes,
        (*proof_bytes).bytes,
        &settings,
    ));

    *ok = result;
    CKzgRet::Ok
}

/// # Safety
#[no_mangle]
#[cfg(feature = "c_bindings")]
pub unsafe extern "C" fn verify_blob_kzg_proof_batch(
    ok: *mut bool,
    blobs: *const Blob,
    commitments_bytes: *const Bytes48,
    proofs_bytes: *const Bytes48,
    n: usize,
    s: &CKZGSettings,
) -> CKzgRet {
    use kzg::eip_4844::verify_blob_kzg_proof_batch_raw;

    let raw_blobs = core::slice::from_raw_parts(blobs, n)
        .iter()
        .map(|blob| blob.bytes)
        .collect::<Vec<_>>();
    let raw_commitments = core::slice::from_raw_parts(commitments_bytes, n)
        .iter()
        .map(|c| c.bytes)
        .collect::<Vec<_>>();
    let raw_proofs = core::slice::from_raw_parts(proofs_bytes, n)
        .iter()
        .map(|p| p.bytes)
        .collect::<Vec<_>>();

    *ok = false;

    let settings: MclKZGSettings = handle_ckzg_badargs!(s.try_into());
    let result = handle_ckzg_badargs!(verify_blob_kzg_proof_batch_raw(
        &raw_blobs,
        &raw_commitments,
        &raw_proofs,
        &settings
    ));

    *ok = result;

    CKzgRet::Ok
}

/// # Safety
#[no_mangle]
#[cfg(feature = "c_bindings")]
pub unsafe extern "C" fn compute_kzg_proof(
    proof_out: *mut KZGProof,
    y_out: *mut Bytes32,
    blob: *const Blob,
    z_bytes: *const Bytes32,
    s: &CKZGSettings,
) -> CKzgRet {
    use kzg::eip_4844::compute_kzg_proof_raw;

    let settings: MclKZGSettings = handle_ckzg_badargs!(s.try_into());

    let (proof_out_tmp, fry_tmp) = handle_ckzg_badargs!(compute_kzg_proof_raw(
        (*blob).bytes,
        (*z_bytes).bytes,
        &settings
    ));

    (*proof_out).bytes = proof_out_tmp.to_bytes();
    (*y_out).bytes = fry_tmp.to_bytes();
    CKzgRet::Ok
}
