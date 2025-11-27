use crate::{
    msm::pippenger_utils::{booth_encode, get_wval_limb, is_zero},
    Fr, G1Affine, G1Fp, G1GetFp, G1Mul, G1ProjAddAffine, Scalar256, G1,
};
use core::cmp;
use std::marker::PhantomData;

const NBITS: usize = 255;

#[derive(Debug, Clone)]
pub struct WbitsTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine> {
    numpoints: usize,
    points: Vec<TG1Affine>,

    // batch_numpoints: usize,
    // batch_points: Vec<Vec<TG1Affine>>,
    g1_marker: PhantomData<TG1>,
    g1_fp_marker: PhantomData<TG1Fp>,
    fr_marker: PhantomData<TFr>,
    g1_affine_add_marker: PhantomData<TG1ProjAddAffine>,
}

impl<
        TFr: Fr,
        TG1Fp: G1Fp,
        TG1: G1 + G1Mul<TFr> + G1GetFp<TG1Fp>,
        TG1Affine: G1Affine<TG1, TG1Fp>,
        TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
    > WbitsTable<TFr, TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>
{
    pub fn new(points: &[TG1], matrix: &[Vec<TG1>]) -> Result<Option<Self>, String> {
        let points = points
            .iter()
            .map(TG1Affine::into_affine)
            .collect::<Vec<_>>();

        let output = wbits_precompute::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(&points, 8);

        Ok(Some(Self {
            numpoints: points.len(),
            points: output,

            g1_marker: PhantomData,
            g1_fp_marker: PhantomData,
            fr_marker: PhantomData,
            g1_affine_add_marker: PhantomData,
        }))
    }

    pub fn multiply_sequential(&self, scalars: &[TFr]) -> TG1 {
        let scalars = scalars.iter().map(TFr::to_scalar).collect::<Vec<_>>();

        mult_wbits::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(&self.points, 8, &scalars)
    }

    // pub fn multiply_batch() {
    //     let mut result = Vec::new();

    //     for (points, scalars) in points.iter().zip(scalars.iter()) {
    //         if points.len() != scalars.len() {
    //             return Err("Invalid point count length".to_owned());
    //         }

    //         result.push(Self::g1_lincomb(points, scalars, points.len(), None));
    //     }

    //     Ok(result)
    // }
}

fn gather_booth_wbits<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    row: &[TG1Affine],
    wbits: usize,
    booth_idx: usize,
) -> TG1Affine {
    let booth_sign = (booth_idx >> wbits) & 1;

    let booth_idx = booth_idx & ((1usize << wbits) - 1);
    let idx_is_zero = is_zero(booth_idx as u64);
    let booth_idx = booth_idx as u64 - (1 ^ idx_is_zero);

    if idx_is_zero == 1 {
        TG1Affine::zero()
    } else {
        if booth_sign == 1 {
            row[booth_idx as usize].neg()
        } else {
            row[booth_idx as usize]
        }
    }
}

fn head<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    ab: &mut [(TG1Affine, TG1Fp)],
    mul_acc: Option<&TG1Fp>,
) {
    let (a, b) = ab.split_at_mut(1);

    let mut inf = a[0].0.is_zero() || b[0].0.is_zero();

    b[0].1 = b[0].0.x().sub_fp(a[0].0.x()); // X2-X1
    b[0].0.x_mut().add_assign_fp(a[0].0.x()); // X2+X1
    a[0].1 = b[0].0.y().add_fp(a[0].0.y()); // Y2+Y1
    b[0].0.y_mut().sub_assign_fp(a[0].0.y()); // Y2-Y1

    if b[0].1.is_zero() {
        // X2==X1
        inf = a[0].1.is_zero();
        *b[0].0.x_mut() = if inf { a[0].1 } else { b[0].0.x().clone() };
        *b[0].0.y_mut() = a[0].0.x().square();
        *b[0].0.y_mut() = b[0].0.y().mul3(); // 3*X1^2
        b[0].1 = a[0].1; // 2*Y1
    } // b.0.y is numenator
      // b.1 is denominator

    *a[0].0.x_mut() = if inf {
        b[0].0.x().clone()
    } else {
        a[0].0.x().clone()
    };
    *a[0].0.y_mut() = if inf { a[0].1 } else { a[0].0.y().clone() };
    a[0].1 = if inf { TG1Fp::one() } else { b[0].1 };
    b[0].1 = if inf { TG1Fp::zero() } else { b[0].1 };
    if let Some(mul_acc) = mul_acc {
        a[0].1.mul_assign_fp(mul_acc);
    }
}

fn tail<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    a: &mut TG1Affine,
    b: &mut (TG1Affine, TG1Fp),
    lambda: &mut TG1Fp,
) -> TG1Affine {
    let mut dest = TG1Affine::zero();

    let inf = b.1.is_zero();

    lambda.mul_assign_fp(b.0.y()); // λ = (Y2-Y1)/(X2-X1)

    let llambda = lambda.square();
    *dest.x_mut() = llambda.sub_fp(b.0.x()); // X3 = λ^2-X1-X2

    *dest.y_mut() = a.x().sub_fp(dest.x());
    dest.y_mut().mul_assign_fp(&lambda);
    dest.y_mut().sub_assign_fp(a.y()); // Y3 = λ*(X1-X3)-Y1

    if inf {
        *dest.x_mut() = a.x().clone();
        *dest.y_mut() = a.y().clone();
    }

    b.1 = if inf { TG1Fp::one() } else { b.1 };

    dest
}

fn accumulate<
    TG1: G1,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
>(
    sum: &mut TG1,
    mut points: &mut [(TG1Affine, TG1Fp)],
) {
    let mut n = points.len();

    while n >= 16 {
        if (n & 1) == 1 {
            TG1ProjAddAffine::add_assign_affine(sum, &points[0].0);
            points = &mut points[1..];
            n -= 1;
        }

        let mut mul_acc = None;
        for i in (0..n).step_by(2) {
            head(&mut points[i..(i + 2)], mul_acc.as_ref());
            mul_acc = Some(points[i].1);
        }

        points[n - 2].1 = points[n - 2].1.inverse().unwrap();

        for i in (2..(n - 1)).rev().step_by(2) {
            let dest = i / 2 + n / 2;
            points[i - 2].1 = points[i - 2].1.mul_fp(&points[i].1);
            points[dest].0 = {
                let mut dest = TG1Affine::zero();

                let inf = points[i + 1].1.is_zero();

                points[i - 2].1 = points[i - 2].1.mul_fp(points[i + 1].0.y()); // λ = (Y2-Y1)/(X2-X1)

                let llambda = points[i - 2].1.square();
                *dest.x_mut() = llambda.sub_fp(points[i + 1].0.x()); // X3 = λ^2-X1-X2

                *dest.y_mut() = points[i].0.x().sub_fp(dest.x());
                dest.y_mut().mul_assign_fp(&points[i - 2].1);
                dest.y_mut().sub_assign_fp(points[i].0.y()); // Y3 = λ*(X1-X3)-Y1

                if inf {
                    *dest.x_mut() = points[i].0.x().clone();
                    *dest.y_mut() = points[i].0.y().clone();
                }

                points[i + 1].1 = if inf { TG1Fp::one() } else { points[i + 1].1 };

                dest
            };
            points[i - 2].1 = points[i].1.mul_fp(&points[i + 1].1);
        }

        points[n / 2].0 = {
            let mut dest = TG1Affine::zero();

            let inf = points[1].1.is_zero();

            points[0].1.mul_assign_fp(points[1].0.y()); // λ = (Y2-Y1)/(X2-X1)

            let llambda = points[0].1.square();
            *dest.x_mut() = llambda.sub_fp(points[1].0.x()); // X3 = λ^2-X1-X2

            *dest.y_mut() = points[0].0.x().sub_fp(dest.x());
            dest.y_mut().mul_assign_fp(&points[0].1);
            dest.y_mut().sub_assign_fp(points[0].0.y()); // Y3 = λ*(X1-X3)-Y1

            if inf {
                *dest.x_mut() = points[0].0.x().clone();
                *dest.y_mut() = points[0].0.y().clone();
            }

            points[1].1 = if inf { TG1Fp::one() } else { points[1].1 };

            dest
        };
        points = &mut points[n / 2..];
        n /= 2;
    }

    for i in 0..n {
        TG1ProjAddAffine::add_assign_affine(sum, &points[i].0);
    }
}

fn mult_wbits<
    TG1: G1,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
>(
    table: &[TG1Affine],
    wbits: usize,
    scalars: &[Scalar256],
) -> TG1 {
    let npoints = scalars.len();
    let scratch_sz = cmp::min(8192, npoints);
    let mut scratch = vec![(TG1Affine::zero(), TG1Fp::zero()); scratch_sz];

    let nwin = 1usize << (wbits - 1);
    let scalar = scalars[0];

    let mut window = NBITS % wbits;
    let mut wmask = (1u64 << (window + 1)) - 1;

    let mut nbits = NBITS - window;
    let z = is_zero(nbits as u64);
    let wval = (get_wval_limb(&scalar, nbits - (z as usize ^ 1), window + (z as usize ^ 1))
        << z as usize)
        & wmask;
    let wval = booth_encode(wval, wbits);
    scratch[0].0 = gather_booth_wbits(table, wbits, wval as usize);

    let mut result = TG1::zero();

    let mut start_i = 1;
    while nbits > 0 {
        let mut j = start_i;
        for i in start_i..npoints {
            if j == scratch_sz {
                accumulate::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(&mut result, &mut scratch);
                j = 0;
            }

            let scalar = scalars[i];
            let wval = get_wval_limb(&scalar, nbits - 1, window + 1) & wmask;
            let wval = booth_encode(wval, wbits);
            scratch[j].0 = gather_booth_wbits(&table[i * nwin..], wbits, wval as usize);
            j += 1;
        }
        accumulate::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(&mut result, &mut scratch[0..j]);

        for _ in 0..wbits {
            result.dbl_assign();
        }

        window = wbits;
        wmask = (1u64 << (window + 1)) - 1;
        nbits -= window;
        start_i = 0;
    }

    let mut j = start_i;
    for i in start_i..npoints {
        if j == scratch_sz {
            accumulate::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(&mut result, &mut scratch);
            j = 0;
        }
        let scalar = scalars[i];
        let wval = (get_wval_limb(&scalar, 0, window) << 1) & wmask;
        let wval = booth_encode(wval, wbits);
        scratch[j].0 = gather_booth_wbits(&table[i * nwin..], wbits, wval as usize);
        j += 1;
    }
    accumulate::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(&mut result, &mut scratch[..j]);

    result
}

fn wbits_precompute<
    TG1: G1 + G1GetFp<TG1Fp>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
>(
    points: &[TG1Affine],
    wbits: usize,
) -> Vec<TG1Affine> {
    let total = points.len() << (wbits - 1);
    let mut table = vec![TG1Affine::zero(); total];
    let nwin = 1usize << (wbits - 1);
    let nmin = if wbits > 9 {
        1usize
    } else {
        1usize << (9 - wbits)
    };

    let mut stride = ((512 * 1024) / size_of::<TG1Affine>()) >> wbits;
    if stride == 0 {
        stride = 1;
    }

    let mut top = 0;
    let mut npoints = points.len();
    let mut pi = 0;

    let mut rows = Vec::with_capacity(total);
    while npoints >= nmin {
        let limit = total - npoints;
        if top + (stride << wbits) > limit {
            stride = (limit - top) >> wbits;
            if stride == 0 {
                break;
            }
        }

        rows.clear();
        for i in 0..stride {
            precompute_row_wbits::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(
                &mut rows,
                wbits,
                &points[pi],
            );
            pi += 1;
        }
        to_affine_row_bits(&mut table[top..], &rows, wbits, stride);
        top += stride << (wbits - 1);
        npoints -= stride;
    }

    rows.clear();
    for i in 0..npoints {
        precompute_row_wbits::<TG1, TG1Fp, TG1Affine, TG1ProjAddAffine>(
            &mut rows,
            wbits,
            &points[pi],
        );
        pi += 1;
    }
    to_affine_row_bits(&mut table[top..], &rows, wbits, npoints);

    table
}

fn precompute_row_wbits<
    TG1: G1 + G1GetFp<TG1Fp>,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
>(
    points: &mut Vec<TG1>,
    wbits: usize,
    point: &TG1Affine,
) {
    let n = 1usize << (wbits - 1);
    let inf = point.is_infinity();

    points.push(point.to_proj());
    points.push(points[0].dbl());

    if inf {
        *points[1].z_mut() = TG1Fp::one();
    }

    for j in 1..(n / 2) {
        points.push(TG1ProjAddAffine::add_or_double_affine(
            points[j * 2 - 1].clone(),
            &point,
        ));
        points.push(points[j].dbl());
        if inf {
            *points.last_mut().unwrap().z_mut() = TG1Fp::one();
        }
    }
}

fn to_affine_row_bits<TG1: G1 + G1GetFp<TG1Fp>, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    dst: &mut [TG1Affine],
    src: &[TG1],
    wbits: usize,
    npoints: usize,
) {
    let total = npoints << (wbits - 1);
    let nwin = 1usize << (wbits - 1);

    let mut acc = vec![TG1Fp::zero(); total + 1];
    acc[0] = TG1Fp::one();

    let mut acc_ptr = 1;
    let mut src_ptr = total;
    for i in 0..npoints {
        for j in (0..nwin).rev() {
            src_ptr -= 1;
            acc[acc_ptr] = acc[acc_ptr - 1].mul_fp(src[src_ptr].z());
            acc_ptr += 1;
        }
    }

    acc_ptr -= 1;
    acc[0] = acc[0].inverse().unwrap();

    let mut dst_ptr = 0;
    for i in 0..npoints {
        *dst[dst_ptr].x_mut() = src[src_ptr].x().clone();
        *dst[dst_ptr].y_mut() = src[src_ptr].y().clone();
        dst_ptr += 1;
        src_ptr += 1;
        for j in 1..nwin {
            acc[acc_ptr - 1] = acc[acc_ptr - 1].mul_fp(&acc[acc_ptr]);
            let zz = acc[acc_ptr - 1].square();
            let zzz = zz.mul_fp(&acc[acc_ptr - 1]);
            acc[acc_ptr - 1] = src[src_ptr].z().mul_fp(&acc[acc_ptr]);
            *dst[dst_ptr].x_mut() = src[src_ptr].x().mul_fp(&zz);
            *dst[dst_ptr].y_mut() = src[src_ptr].y().mul_fp(&zzz);
            acc_ptr -= 1;
            src_ptr += 1;
            dst_ptr += 1;
        }
    }
}
