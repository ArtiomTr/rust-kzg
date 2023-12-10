use blst::{blst_p1, blst_p1_affine, blst_p1_to_affine, limb_t};
use kzg::{Fr, G1Mul};

use crate::{
    msm::pippenger::is_zero,
    types::{fr::FsFr, g1::FsG1},
};

use super::pippenger::{
    booth_encode, get_wval_limb, p1_dadd, p1_to_jacobian, p1s_bucket, pippenger_window_size,
    scalars_to_bytes, P1XYZZ,
};

#[derive(Debug, Clone)]
pub struct BGMWTable {
    pub numpoints: usize,
    pub h: usize,
    pub precomputed: Vec<blst_p1_affine>,
    pub window: BGMWWindow,
}

#[derive(Debug, Clone)]
pub enum BGMWWindow {
    Synchronous(usize),
    Parallel((usize, usize, usize)),
}

impl BGMWTable {
    pub fn compute(points: &[FsG1]) -> Result<Self, String> {
        precompute_bgmw(points)
        // Err("Test".to_string())
    }
}

fn bgmw_window(npoints: usize) -> BGMWWindow {
    let default_window = pippenger_window_size(npoints);

    #[cfg(all(feature = "parallel", feature = "std"))]
    {
        use super::bgmw_parallel::{breakdown, da_pool};
        let pool = da_pool();
        let ncpus = pool.max_count();
        if npoints > 32 && ncpus > 2 {
            return BGMWWindow::Parallel(breakdown(255, default_window, ncpus));
        }
    }

    BGMWWindow::Synchronous(default_window)
}

fn precompute_bgmw(points: &[FsG1]) -> Result<BGMWTable, String> {
    const NBITS: usize = 255;

    let window = bgmw_window(points.len());

    let (window_width, h) = match window {
        BGMWWindow::Parallel((_, ny, wnd)) => (wnd, ny),
        BGMWWindow::Synchronous(wnd) => {
            let h = (NBITS + wnd - 1) / wnd + is_zero((NBITS % wnd) as limb_t) as usize;

            (wnd, h)
        }
    };

    let mut table: Vec<blst_p1_affine> = Vec::new();
    let q = FsFr::from_u64(1u64 << window_width);

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

/// Calculate bucket sum
///
/// This function multiplies the point in each bucket by it's index. Then, it will sum all multiplication results and write
/// resulting point to the `out`.
///
/// This function also clears all buckets (sets all values in buckets to zero.)
///
/// ## Arguments
///
/// * out     - output where bucket sum must be written
/// * buckets - pointer to the beginning of the array of buckets
/// * wbits   - window size, aka exponent of q (q^window)
///  
pub fn p1_integrate_buckets(out: &mut blst_p1, buckets: &[P1XYZZ], wbits: usize) {
    // Calculate total amount of buckets
    let mut n = (1usize << wbits) - 1;

    // Resulting point
    let mut ret = buckets[n];
    // Accumulator
    let mut acc = buckets[n];

    /*
     * Sum all buckets.
     *
     * Starting from the end, this loop adds points to accumulator, and then adds points to the result.
     * If the point is in the bucket `i`, then adding this point to the accumulator and adding accumulator `i` times
     * helps to avoid multiplication of point by `i`.
     *
     * Example:
     *
     * If we have 3 buckets with points [`p1`, `p2`, `p3`], and we need to calculate bucket sum, naive approach would be:
     * `S` = `p1` + 2 * `p2` + 3 * `p3` (which is `p1` + `p2` + `p2` + `p3` + `p3` + `p3` - 5 additions)
     * But using accumulator, it would be:
     *
     * ```rust
     * acc = p3;
     * ret = p3;
     * acc += p2;   // now acc contains `p2` + `p3`
     * ret += acc;  // now res contains `p2` + 2*`p3`
     * acc += p1;   // now acc contains `p1` + `p2` + `p3`
     * ret += acc;  // now res contains `p1` + 2*`p2` + 3*`p3`
     * ```
     *
     * 4 additions. So using accumulator, we saved 1 addition.
     */
    loop {
        if n == 0 {
            break;
        }
        n -= 1;

        // Add point to accumulator
        unsafe { p1_dadd(&mut acc, &acc, &buckets[n]) };
        // Add accumulator to result
        unsafe { p1_dadd(&mut ret, &ret, &acc) };
    }

    // Convert point from magical 4-coordinate system to Jacobian (normal)
    unsafe { p1_to_jacobian(out, &ret) };
}

pub fn bgmw_tile_pub(
    points: &[blst_p1_affine],
    npoints: usize,
    scalars: &[u8],
    nbits: usize,
    buckets: &mut [P1XYZZ],
    bit0: usize,
    window: usize,
) {
    let (wbits, cbits) = if bit0 + window > nbits {
        let wbits = nbits - bit0;
        (wbits, wbits + 1)
    } else {
        (window, window)
    };

    bgmw_tile(points, npoints, scalars, nbits, buckets, bit0, wbits, cbits);
}

#[allow(clippy::too_many_arguments)]
fn bgmw_tile(
    points: &[blst_p1_affine],
    mut npoints: usize,
    scalars: &[u8],
    nbits: usize,
    buckets: &mut [P1XYZZ],
    bit0: usize,
    wbits: usize,
    cbits: usize,
) {
    // Calculate number of bytes, to fit `nbits`. Basically, this is division by 8 with rounding up to nearest integer.
    let nbytes = (nbits + 7) / 8;

    // Get first scalar
    let scalar = &scalars[0..nbytes];

    // Get first point
    let point = &points[0];

    // Create mask, that contains `wbits` amount of ones at the end.
    let wmask = ((1 as limb_t) << (wbits + 1)) - 1;

    /*
     * Check if `bit0` is zero. `z` is set to `1` when `bit0 = 0`, and `0` otherwise.
     *
     * The `z` flag is used to do a small trick -
     */
    let z = is_zero(bit0.try_into().unwrap());

    // Offset `bit0` by 1, if it is not equal to zero.
    let bit0 = bit0 - (z ^ 1) as usize;

    // Increase `wbits` by one, if `bit0` is not equal to zero.
    let wbits = wbits + (z ^ 1) as usize;

    // Calculate first window value (encoded bucket index)
    let wval = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let mut wval = booth_encode(wval, cbits);

    // Get second scalar
    let scalar = &scalars[nbytes..2 * nbytes];

    // Calculate second window value (encoded bucket index)
    let wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
    let mut wnxt = booth_encode(wnxt, cbits);

    // Move first point to corresponding bucket
    p1s_bucket(buckets, wval, cbits, point);

    // Last point will be calculated separately, so decrementing point count
    npoints -= 1;

    // Move points to buckets
    for i in 1..npoints {
        // Get current window value (encoded bucket index)
        wval = wnxt;

        // Get next scalar
        let scalar = &scalars[(i + 1) * nbytes..(i + 2) * nbytes];
        // Get next window value (encoded bucket index)
        wnxt = (get_wval_limb(scalar, bit0, wbits) << z) & wmask;
        wnxt = booth_encode(wnxt, cbits);

        // TODO: add prefetching
        // POINTonE1_prefetch(buckets, wnxt, cbits);
        // p1_prefetch(buckets, wnxt, cbits);

        // Get current point
        let point = &points[i];

        // Move point to corresponding bucket (add or subtract from bucket)
        // `wval` contains encoded bucket index, as well as sign, which shows if point should be subtracted or added to bucket
        p1s_bucket(buckets, wval, cbits, point);
    }
    // Get last point
    let point = &points[npoints];
    // Move point to bucket
    p1s_bucket(buckets, wnxt, cbits, point);
    // Integrate buckets - multiply point in each bucket by scalar and sum all results
    // p1_integrate_buckets(ret, buckets, cbits - 1);
}

fn bgmw_impl(table: &BGMWTable, npoints: usize, scalars: &[u8], buckets: &mut [P1XYZZ]) -> blst_p1 {
    let mut ret = blst_p1::default();

    const NBITS: usize = 255;
    let window = match table.window {
        BGMWWindow::Synchronous(wnd) => wnd,
        BGMWWindow::Parallel((_, _, wnd)) => wnd,
    };

    let mut wbits: usize = NBITS % window;
    let mut cbits: usize = wbits + 1;
    let mut bit0: usize = NBITS;
    let mut q_idx = table.h;

    loop {
        bit0 -= wbits;
        q_idx -= 1;
        if bit0 == 0 {
            break;
        }

        bgmw_tile(
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
        // unsafe { blst_p1_add(&mut ret, &ret, &tile) };

        cbits = window;
        wbits = window;
    }
    bgmw_tile(
        &table.precomputed[0..table.numpoints],
        npoints,
        scalars,
        NBITS,
        buckets,
        0,
        wbits,
        cbits,
    );
    p1_integrate_buckets(&mut ret, buckets, wbits - 1);
    // unsafe { blst_p1_add(&mut ret, &ret, &tile) };

    ret
}

/// Multithread Pippenger's algorithm implementation
#[cfg(all(feature = "parallel", feature = "std"))]
fn bgmw_async(table: &BGMWTable, scalars: &[FsFr]) -> FsG1 {
    use crate::msm::bgmw_parallel::{da_pool, multiply};

    if let BGMWWindow::Parallel(window) = table.window {
        let npoints = scalars.len();

        let scalars = scalars_to_bytes(scalars);

        return FsG1(multiply(table, window, npoints, &scalars, 255, da_pool()));
    }

    bgmw_sync(table, scalars)
}

fn bgmw_sync(table: &BGMWTable, scalars: &[FsFr]) -> FsG1 {
    let npoints = scalars.len();

    // assert!(npoints <= table.numpoints);

    let window = match table.window {
        BGMWWindow::Synchronous(wnd) => wnd,
        BGMWWindow::Parallel((_, _, wnd)) => wnd,
    };

    let mut buckets = vec![P1XYZZ::default(); 1usize << (window - 1)];

    let scalars = scalars_to_bytes(scalars);

    FsG1(bgmw_impl(table, npoints, &scalars, &mut buckets))
}

pub fn bgmw(table: &BGMWTable, scalars: &[FsFr]) -> FsG1 {
    #[cfg(all(feature = "parallel", feature = "std"))]
    return bgmw_async(table, scalars);

    #[cfg(not(all(feature = "parallel", feature = "std")))]
    return bgmw_sync(table, scalars);
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
