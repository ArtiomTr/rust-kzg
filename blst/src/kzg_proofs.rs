extern crate alloc;

#[cfg(not(feature = "parallel"))]
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use blst::{blst_p1, blst_scalar, blst_scalar_from_fr, p1_affines};
#[cfg(not(feature = "parallel"))]
use blst::{blst_p1s_mult_pippenger_scratch_sizeof, limb_t};

use crate::types::kzg_settings::BGMWPreComputationList;
use blst::{
    blst_fp12_is_one, blst_p1_affine, blst_p1_cneg, blst_p1_to_affine, blst_p2_affine,
    blst_p2_to_affine, Pairing,
};

use kzg::{G1Mul, PairingVerify, G1};

#[cfg(not(feature = "parallel"))]
use crate::msm::msm;
use crate::types::fr::FsFr;
use crate::types::g1::FsG1;
use crate::types::g2::FsG2;

impl PairingVerify<FsG1, FsG2> for FsG1 {
    fn verify(a1: &FsG1, a2: &FsG2, b1: &FsG1, b2: &FsG2) -> bool {
        pairings_verify(a1, a2, b1, b2)
    }
}

#[allow(unused)]
pub fn g1_linear_combination(
    out: &mut FsG1,
    points: &[FsG1],
    scalars: &[FsFr],
    len: usize,
    table: Option<&BGMWPreComputationList>,
) {
    if len < 8 {
        *out = FsG1::default();
        for i in 0..len {
            let tmp = points[i].mul(&scalars[i]);
            *out = out.add_or_dbl(&tmp);
        }
        return;
    }

    #[cfg(feature = "parallel")]
    {
        let points = unsafe { core::slice::from_raw_parts(points.as_ptr() as *const blst_p1, len) };
        let points = p1_affines::from(points);
        let mut scalar_bytes: Vec<u8> = Vec::with_capacity(len * 32);
        for bytes in scalars.iter().map(|b| {
            let mut scalar = blst_scalar::default();

            unsafe { blst_scalar_from_fr(&mut scalar, &b.0) }

            scalar.b
        }) {
            scalar_bytes.extend_from_slice(&bytes);
        }

        let res = points.mult(scalar_bytes.as_slice(), 255);
        *out = FsG1(res)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch: Vec<u8>;
        unsafe {
            scratch = vec![0u8; blst_p1s_mult_pippenger_scratch_sizeof(len)];
        }

        msm(
            out,
            points,
            len,
            scalars,
            255,
            scratch.as_mut_ptr() as *mut limb_t,
            table,
        );
    }
}

pub fn pairings_verify(a1: &FsG1, a2: &FsG2, b1: &FsG1, b2: &FsG2) -> bool {
    let mut aa1 = blst_p1_affine::default();
    let mut bb1 = blst_p1_affine::default();

    let mut aa2 = blst_p2_affine::default();
    let mut bb2 = blst_p2_affine::default();

    // As an optimisation, we want to invert one of the pairings,
    // so we negate one of the points.
    let mut a1neg: FsG1 = *a1;
    unsafe {
        blst_p1_cneg(&mut a1neg.0, true);
        blst_p1_to_affine(&mut aa1, &a1neg.0);

        blst_p1_to_affine(&mut bb1, &b1.0);
        blst_p2_to_affine(&mut aa2, &a2.0);
        blst_p2_to_affine(&mut bb2, &b2.0);

        let dst = [0u8; 3];
        let mut pairing_blst = Pairing::new(false, &dst);
        pairing_blst.raw_aggregate(&aa2, &aa1);
        pairing_blst.raw_aggregate(&bb2, &bb1);
        let gt_point = pairing_blst.as_fp12().final_exp();

        blst_fp12_is_one(&gt_point)
    }
}

#[cfg(test)]
mod test {
    use core::ptr;
    use std::{fs::File, io::Read};

    use blst::{blst_p1_affine, blst_p1s_to_affine};
    use kzg::{Fr, G1};

    use crate::{
        bgmw,
        kzg_proofs::g1_linear_combination,
        types::{fr::FsFr, g1::FsG1, kzg_settings::BGMWPreComputationList},
    };

    #[test]
    fn test_test() {
        // {
        //     let len: usize = 128;
        //     let mut coeffs = vec![FsFr::default(); len];
        //     let mut p = vec![FsG1::default(); len];
        //     let mut p1tmp = FsG1::generator();

        //     for i in 0..len {
        //         coeffs[i] = FsFr::rand();
        //         p[i] = p1tmp;
        //         p1tmp = p1tmp.dbl();
        //     }

        //     let mut exp = FsG1::identity();
        //     for i in 0..len {
        //         p1tmp = p[i].mul(&coeffs[i]);
        //         exp = exp.add_or_dbl(&p1tmp);
        //     }

        //     let mut out = String::new();

        //     out += len.to_string().as_str();
        //     out += "\n";

        //     for pt in p.clone() {

        //         out += hex::encode(bytes).as_str();
        //         out += "\n";
        //     }

        //     for c in coeffs.clone() {
        //         out += hex::encode(c.to_bytes()).as_str();
        //         out += "\n";
        //     }

        //     let mut bytes = [0u8; 48];
        //     unsafe {
        //         blst_p1_compress(bytes.as_mut_ptr(), &exp.0);
        //     }
        //     out += hex::encode(bytes).as_str();
        //     out += "\n";

        //     let mut test_f = File::create("test.txt").unwrap();
        //     test_f.write_all(out.as_bytes()).unwrap();
        // }

        let mut test_file = File::open("test.txt").unwrap();
        let mut contents = String::new();
        test_file.read_to_string(&mut contents).unwrap();

        let mut lines = contents.lines();

        let len = lines.next().unwrap().parse::<usize>().unwrap();
        let p = (0..len)
            .map(|_| FsG1::from_bytes(&hex::decode(lines.next().unwrap()).unwrap()).unwrap())
            .collect::<Vec<FsG1>>();
        let coeffs = (0..len)
            .map(|_| FsFr::from_bytes(&hex::decode(lines.next().unwrap()).unwrap()).unwrap())
            .collect::<Vec<FsFr>>();
        let exp = FsG1::from_bytes(&hex::decode(lines.next().unwrap()).unwrap()).unwrap();

        let mut res = FsG1::default();

        let mut p_affine = vec![blst_p1_affine::default(); len];
        let points = [&p[0].0, ptr::null()];

        unsafe {
            blst_p1s_to_affine(p_affine.as_mut_ptr(), points.as_ptr(), len);
        };

        let mut table = vec![blst::blst_p1_affine::default(); bgmw::get_bgmw_table_size(len)];

        bgmw::init_pippenger_bgmw(&mut table, &p_affine);

        g1_linear_combination(
            &mut res,
            &p,
            &coeffs,
            len,
            Some(&BGMWPreComputationList(table)),
        );

        assert!(exp.equals(&res));
    }
}
