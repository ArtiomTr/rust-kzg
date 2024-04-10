
use std::{fs::File, io::Read, path::Path, ptr};

use blst::{blst_p1, blst_p1_affine, blst_p1_from_affine, blst_p1_uncompress, blst_p2, blst_p2_affine, blst_p2_from_affine, blst_p2_uncompress, BLST_ERROR};
use kzg::{common_utils::reverse_bit_order, eip_4844::{
    Blob, Bytes32, Bytes48, CKZGSettings, KZGCommitment, KZGProof, BYTES_PER_FIELD_ELEMENT,
    BYTES_PER_G1, BYTES_PER_G2, C_KZG_RET, C_KZG_RET_BADARGS, C_KZG_RET_OK,
    FIELD_ELEMENTS_PER_BLOB, TRUSTED_SETUP_NUM_G1_POINTS, TRUSTED_SETUP_NUM_G2_POINTS,
}};

pub fn blob_to_kzg_commitment(blob: &Blob, settings: &CKZGSettings) -> KZGCommitment {
    #[cfg_attr(feature = "quiet", allow(improper_ctypes))]
    extern "C" {
        fn sppark_blob_to_kzg_commitment(
            out: *mut KZGCommitment,
            blob: *const Blob,
            s: &CKZGSettings,
        ) -> ();
    }

    let mut commitment = KZGCommitment {
        bytes: [0u8; 48]
    };
    unsafe {
        sppark_blob_to_kzg_commitment(&mut commitment, blob, settings)
    }

    commitment
}

pub fn load_trusted_setup(g1_bytes: &[u8], g2_bytes: &[u8]) -> CKZGSettings {
    let mut g1_values = g1_bytes.chunks_exact(BYTES_PER_G1).map(|bytes| {
        let mut tmp = blst_p1_affine::default();
        let mut g1 = blst_p1::default();
        unsafe {
            // The uncompress routine also checks that the point is on the curve
            if blst_p1_uncompress(&mut tmp, bytes.as_ptr()) != BLST_ERROR::BLST_SUCCESS {
                return Err("Failed to uncompress".to_string());
            }
            blst_p1_from_affine(&mut g1, &tmp);
        }
        Ok(g1)
    }).collect::<Result<Vec<_>, _>>().unwrap();

    let g2_values = g2_bytes.chunks_exact(BYTES_PER_G2).map(|bytes| {
        let mut tmp = blst_p2_affine::default();
        let mut g2 = blst_p2::default();
        unsafe {
            // The uncompress routine also checks that the point is on the curve
            if blst_p2_uncompress(&mut tmp, bytes.as_ptr()) != BLST_ERROR::BLST_SUCCESS {
                return Err("Failed to uncompress".to_string());
            }
            blst_p2_from_affine(&mut g2, &tmp);
        }
        Ok(g2)
    }).collect::<Result<Vec<_>, _>>().unwrap();

    let mut max_scale: usize = 0;
    while (1 << max_scale) < g1_values.len() {
        max_scale += 1;
    };

    reverse_bit_order(&mut g1_values).unwrap();

    CKZGSettings { max_width: max_scale as u64, roots_of_unity: ptr::null_mut(), g1_values: Box::leak(g1_values.into_boxed_slice()).as_mut_ptr(), g2_values: Box::leak(g2_values.into_boxed_slice()).as_mut_ptr() }
}

pub fn load_trusted_setup_file(path: impl AsRef<Path>) -> CKZGSettings {
    let mut file = File::open(path).unwrap();

    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();

    let (g1_bytes, g2_bytes) = kzg::eip_4844::load_trusted_setup_string(buf.as_str()).unwrap();

    load_trusted_setup(&g1_bytes, &g2_bytes)
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Read, path::PathBuf, ptr};

    use kzg::eip_4844::{Blob, CKZGSettings, KZGCommitment, FIELD_ELEMENTS_PER_BLOB, TRUSTED_SETUP_NUM_G2_POINTS};

    use crate::{blob_to_kzg_commitment, load_trusted_setup_file};

    #[test]
    fn should_create_dummy_commitment() {
        let mut rng = rand::thread_rng();
        let blob_bytes = kzg_bench::tests::eip_4844::generate_random_blob_bytes(&mut rng);

        let expected_commitment = {
            let mut settings = CKZGSettings {
                g1_values: ptr::null_mut(),
                g2_values: ptr::null_mut(),
                max_width: 0,
                roots_of_unity: ptr::null_mut()
            };
            let mut file = File::open(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kzg-bench/src/trusted_setup.txt")).unwrap();
            let mut buf = String::new();
            file.read_to_string(&mut buf).unwrap();
            let (g1_bytes, g2_bytes) = kzg::eip_4844::load_trusted_setup_string(buf.as_str()).unwrap();
            unsafe { rust_kzg_blst::eip_4844::load_trusted_setup(&mut settings, g1_bytes.as_ptr(), FIELD_ELEMENTS_PER_BLOB, g2_bytes.as_ptr(), TRUSTED_SETUP_NUM_G2_POINTS); };

            let mut commitment = KZGCommitment {
                bytes: [0u8; 48]
            };
            unsafe { rust_kzg_blst::eip_4844::blob_to_kzg_commitment(&mut commitment, &Blob { bytes: blob_bytes }, &settings); };

            commitment
        };

        let received_commitment = {
            let settings = load_trusted_setup_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kzg-bench/src/trusted_setup.txt"));
            blob_to_kzg_commitment(&Blob { bytes: blob_bytes }, &settings)
        };

        assert!(expected_commitment.bytes.iter().zip(received_commitment.bytes.iter()).all(|(a, b)| a == b))
    }
}
