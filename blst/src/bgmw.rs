use blst::{
    blst_fp, blst_p1, blst_p1_add_or_double, blst_p1_affine, blst_p1_from_affine, blst_p1_to_affine,
};

pub const H_BGMW95: usize = 22;
pub const H_LEN_SCALAR: usize = 20;
pub const EXPONENT_OF_Q: usize = 13;
pub const Q_RADIX: usize = 1 << EXPONENT_OF_Q;
pub const EXPONENT_OF_Q_BGMW95: usize = 12;
pub const Q_RADIX_PIPPENGER_VARIANT: usize = 1 << EXPONENT_OF_Q_BGMW95;

pub fn single_scalar_multiplication(mut scalar: usize, q: blst_p1_affine) -> blst_p1_affine {
    let mut ret: blst_p1_affine = blst_p1_affine::default();
    let mut inf: blst_p1 = blst_p1 {
        x: blst_fp {
            l: [0, 1, 0, 0, 0, 0],
        },
        y: blst_fp {
            l: [0, 0, 0, 0, 0, 0],
        },
        z: blst_fp {
            l: [0, 0, 0, 0, 0, 0],
        },
    };
    let mut xyz_q: blst_p1 = blst_p1::default();
    unsafe {
        blst_p1_from_affine(&mut xyz_q, &q);
        while scalar > 0 {
            if scalar & 1 != 0 {
                blst_p1_add_or_double(&mut inf, &inf, &xyz_q);
            }

            blst_p1_add_or_double(&mut xyz_q, &xyz_q, &xyz_q);
            scalar >>= 1
        }
        blst_p1_to_affine(&mut ret, &inf);
        ret
    }
}

pub fn get_bgmw_table_size(npoints: usize) -> usize {
    npoints * H_BGMW95
}

pub fn init_pippenger_bgmw(table: &mut [blst_p1_affine], points: &[blst_p1_affine]) {
    if Q_RADIX == Q_RADIX_PIPPENGER_VARIANT {
        // 3nh list
        let mut precomputation_points_list_3nh =
            vec![blst_p1_affine::default(); 3 * points.len() * H_LEN_SCALAR];

        for i in 0..points.len() {
            let mut tmp_p_affine = points[i];
            for j in 0..H_LEN_SCALAR {
                for m in 1..3 {
                    let idx_i_j_m: usize = 3 * (i * H_LEN_SCALAR + j) + m - 1;
                    if m == 1 {
                        precomputation_points_list_3nh[idx_i_j_m] = tmp_p_affine;
                    } else {
                        precomputation_points_list_3nh[idx_i_j_m] =
                            single_scalar_multiplication(m, tmp_p_affine);
                    }
                    tmp_p_affine = single_scalar_multiplication(Q_RADIX, tmp_p_affine);
                }
            }
        }

        for i in 0..points.len() {
            for j in 0..H_BGMW95 {
                let idx = i * H_BGMW95 + j;
                table[idx] = precomputation_points_list_3nh[3 * idx];
            }
        }
    } else {
        for i in 0..points.len() {
            let mut tmp_p_affine = points[i];
            for j in 0..H_BGMW95 {
                let idx = i * H_BGMW95 + j;
                table[idx] = tmp_p_affine;
                tmp_p_affine =
                    single_scalar_multiplication(Q_RADIX_PIPPENGER_VARIANT, tmp_p_affine);
            }
        }
    }
}
