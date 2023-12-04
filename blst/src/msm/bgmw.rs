use blst::{blst_p1, blst_p1_add, blst_p1_affine, blst_p1_to_affine, limb_t};
use kzg::{Fr, G1Mul};

use crate::{
    msm::pippenger::{is_zero, pippenger_tile},
    types::{fr::FsFr, g1::FsG1},
};

use super::pippenger::{pippenger_window_size, scalars_to_bytes, P1XYZZ};

#[derive(Debug, Clone, Default)]
pub struct BGMWTable {
    numpoints: usize,
    h: usize,
    precomputed: Vec<blst_p1_affine>,
    window: usize,
}

impl BGMWTable {
    pub fn compute(points: &[FsG1]) -> Result<Self, String> {
        precompute_bgmw(points)
        // Err("Test".to_string())
    }
}

pub fn bgmw_window_size(npoints: usize) -> usize {
    pippenger_window_size(npoints)
}

fn precompute_bgmw(points: &[FsG1]) -> Result<BGMWTable, String> {
    const NBITS: usize = 255;

    let window = bgmw_window_size(points.len());
    let mut table: Vec<blst_p1_affine> = Vec::new();
    let h = (NBITS + window - 1) / window + is_zero((NBITS % window) as limb_t) as usize;
    let q = FsFr::from_u64(1u64 << window);

    table
        .try_reserve_exact(points.len() * h)
        .map_err(|_| "BGMW precomputation table is too large".to_string())?;

    unsafe { table.set_len(points.len() * h) };

    for i in 0..points.len() {
        let mut tmp_point = points[i];
        for j in 0..h {
            let idx = j * points.len() + i;
            unsafe { blst_p1_to_affine(&mut table[idx], &tmp_point.0) };
            tmp_point = tmp_point.mul(&q);
        }
    }

    table.extend(points.iter().flat_map(|point| {
        let mut tmp_point = *point;

        (0..h).map(move |_| {
            let mut affine = blst_p1_affine::default();
            unsafe { blst_p1_to_affine(&mut affine, &tmp_point.0) };
            tmp_point = tmp_point.mul(&q);
            affine
        })
    }));

    Ok(BGMWTable {
        numpoints: points.len(),
        precomputed: table,
        window,
        h,
    })
}

fn bgmw_impl(table: &BGMWTable, npoints: usize, scalars: &[u8], buckets: &mut [P1XYZZ]) -> blst_p1 {
    let mut ret = blst_p1::default();

    const NBITS: usize = 255;

    let mut wbits: usize = NBITS % table.window;
    let mut cbits: usize = wbits + 1;
    let mut bit0: usize = NBITS;
    let mut tile = blst_p1::default();
    let mut q_idx = table.h;

    loop {
        bit0 -= wbits;
        q_idx -= 1;
        if bit0 == 0 {
            break;
        }

        pippenger_tile(
            &mut tile,
            &table.precomputed[q_idx * table.numpoints..(q_idx + 1) * table.numpoints],
            npoints,
            scalars,
            NBITS,
            buckets,
            bit0,
            wbits,
            cbits,
        );

        // add bucket sum (aka tile) to the return value
        unsafe { blst_p1_add(&mut ret, &ret, &tile) };

        cbits = table.window;
        wbits = table.window;
    }
    pippenger_tile(
        &mut tile,
        &table.precomputed[0..table.numpoints],
        npoints,
        scalars,
        NBITS,
        buckets,
        0,
        wbits,
        cbits,
    );
    unsafe { blst_p1_add(&mut ret, &ret, &tile) };

    ret
}

pub fn bgmw(table: &BGMWTable, scalars: &[FsFr]) -> FsG1 {
    let mut buckets = vec![P1XYZZ::default(); 1usize << (table.window - 1)];

    let npoints = scalars.len();

    // assert!(npoints <= table.numpoints);

    let scalars = scalars_to_bytes(scalars);

    FsG1(bgmw_impl(table, npoints, &scalars, &mut buckets))
}

#[cfg(test)]
mod tests {
    use kzg::{Fr, G1Mul, G1};

    use crate::{
        msm::{bgmw, BGMWTable},
        types::{fr::FsFr, g1::FsG1},
    };

    #[test]
    fn bgmw_scalar_length_divisible_by_window() {
        let npoints = 1usize << 7;

        let points = Vec::from_iter((0..npoints).map(|_| FsG1::rand()));
        let scalars = Vec::from_iter((0..npoints).map(|_| FsFr::rand()));

        let expected = {
            let mut res = FsG1::default();
            for i in 0..npoints {
                res = res.add_or_dbl(&points[i].mul(&scalars[i]))
            }
            res
        };

        let table = BGMWTable::compute(&points).unwrap();

        let received = bgmw::bgmw(&table, &scalars);

        assert_eq!(expected, received);
    }
    //     use std::iter::zip;
    //     use std::path::PathBuf;
    //     use std::{fs::File, io::Read};

    //     use blst::{blst_p1_affine, BLST_ERROR};
    //     use kzg::G1;

    //     use crate::msm::bgmw::precompute_bgmw;
    //     use crate::types::g1::FsG1;

    //     fn get_fixture_path(fixture: &str) -> PathBuf {
    //         PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    //             .join("tests/fixtures")
    //             .join(fixture)
    //             .join("data.txt")
    //     }

    //     #[test]
    //     fn bgmw_precomputation_test() {
    //         struct Fixture {
    //             name: String,
    //             message: String,
    //         }

    //         let fixtures = [
    //             Fixture {
    //                 name: "valid_random_points_bgmw_precomputation".to_string(),
    //                 message: "Precomputing small random point set".to_string(),
    //             },
    //             Fixture {
    //                 name: "valid_p1_generator_bgmw_precomputation".to_string(),
    //                 message: "Precomputing random point set".to_string(),
    //             },
    //             Fixture {
    //                 name: "valid_trusted_setup_bgmw_precomputation".to_string(),
    //                 message: "Precomputing trusted setup points".to_string(),
    //             },
    //         ];

    //         for fixture in fixtures {
    //             let file_path = get_fixture_path(&fixture.name);
    //             let mut file = File::open(file_path.clone()).unwrap();

    //             let mut contents = String::new();

    //             file.read_to_string(&mut contents).unwrap();

    //             let mut lines = contents.lines();

    //             let point_count = lines.next().unwrap().parse::<usize>().unwrap();
    //             let table_size = lines.next().unwrap().parse::<usize>().unwrap();

    //             let points = (0..point_count)
    //                 .map(|_| {
    //                     let bytes = hex::decode(lines.next().unwrap()).unwrap();

    //                     FsG1::from_bytes(&bytes).unwrap()
    //                 })
    //                 .collect::<Vec<FsG1>>();

    //             let expected_table = (0..table_size)
    //                 .map(|_| {
    //                     let bytes = hex::decode(lines.next().unwrap()).unwrap();

    //                     let mut output = blst_p1_affine::default();

    //                     let code = unsafe { blst::blst_p1_uncompress(&mut output, bytes.as_ptr()) };

    //                     assert_eq!(code, BLST_ERROR::BLST_SUCCESS);

    //                     output
    //                 })
    //                 .collect::<Vec<blst_p1_affine>>();

    //             let received_table = precompute_bgmw(&points).unwrap();

    //             assert_eq!(received_table.precomputed.len(), expected_table.len());

    //             for (i, (expected, received)) in
    //                 zip(expected_table, received_table.precomputed).enumerate()
    //             {
    //                 assert_eq!(
    //                     expected,
    //                     received,
    //                     "{}, points at index {} are not equal, fixture = {}",
    //                     fixture.message,
    //                     i,
    //                     file_path.as_os_str().to_str().unwrap()
    //                 );
    //             }
    //         }
    //     }
}
