use crate::{G1Affine, G1Fp, G1ProjAddAffine, G1};

/// This is the threshold to which batching the inversions in affine
/// formula costs more than doing mixed addition.
const BATCH_INVERSE_THRESHOLD: usize = 16;

/// Chooses between point addition and point doubling based on the input points.
#[inline(always)]
fn choose_add_or_double<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    pair: &mut [TG1Affine],
) -> TG1Fp {
    let fp = if pair[0].is_infinity() || pair[1].is_infinity() {
        return TG1Fp::one();
    } else if pair[0].x() == pair[1].x() {
        if pair[0].y() != pair[1].y() {
            // pair[0] == -pair[1]: we mark the pair as neutral so point_add_double is a no-op.
            pair[1] = TG1Affine::zero();
            pair[0] = TG1Affine::zero();
            return TG1Fp::one();
        }

        let f = pair[1].y().double();
        *pair[1].y_mut() = pair[0].x().square().mul3();
        f
    } else {
        *pair[1].y_mut() = pair[1].y().sub_fp(pair[0].y());
        pair[1].x().sub_fp(pair[0].x())
    };

    fp
}

/// Adds two elliptic curve points using the point addition/doubling formula.
///
/// Note: The inversion is precomputed and passed as a parameter.
///
/// This function handles both addition of distinct points and point doubling.
#[inline(always)]
fn point_add_double<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
    p1: TG1Affine,
    p2: TG1Affine,
    inv: &TG1Fp,
) -> TG1Affine {
    if p1.is_zero() {
        return p2;
    }

    if p2.is_zero() {
        return p1;
    }

    let lambda = p2.y().mul_fp(inv);

    let x = lambda.square().sub_fp(p1.x()).sub_fp(p2.x());
    let y = lambda.mul_fp(&p1.x().sub_fp(&x)).sub_fp(p1.y());

    TG1Affine::from_xy(x, y)
}

/// Given a vector of field elements {v_i}, compute the vector {v_i^(-1)}
///
/// A scratchpad is used to avoid excessive allocations in the case that this method is
/// called repeatedly.
///
/// Panics if any of the elements are zero
pub fn batch_inverse_scratch_pad<F: G1Fp>(v: &mut [F], scratchpad: &mut Vec<F>) {
    if v.is_empty() {
        return;
    }

    // Montgomery's Trick and Fast Implementation of Masked AES
    // Genelle, Prouff and Quisquater
    // Section 3.2
    // but with an optimization to multiply every element in the returned vector by coeff

    // Clear the scratchpad and ensure it has enough capacity
    scratchpad.clear();
    scratchpad.reserve(v.len());

    // First pass: compute [a, ab, abc, ...]
    let mut tmp = F::one();
    for f in v.iter() {
        tmp = tmp.mul_fp(f);
        scratchpad.push(tmp);
    }

    // Invert `tmp`.
    tmp = tmp
        .inverse()
        .expect("guaranteed to be non-zero since we filtered out zero field elements");

    // Second pass: iterate backwards to compute inverses
    for i in (1..v.len()).rev() {
        // tmp := tmp * v[i]; v[i] := tmp * scratchpad[i - 1] = 1 / v[i]
        let new_tmp = tmp.mul_fp(&v[i]);
        v[i] = tmp.mul_fp(&scratchpad[i - 1]);
        tmp = new_tmp;
    }

    v[0] = tmp;
}

/// Performs multi-batch addition of multiple sets of elliptic curve points.
///
/// This function efficiently adds multiple sets of points amortizing the cost of the
/// inversion over all of the sets, using the same binary tree approach with striding
/// as the single-batch version.
pub fn multi_batch_addition_binary_tree_stride<
    TG1: G1,
    TG1Fp: G1Fp,
    TG1Affine: G1Affine<TG1, TG1Fp>,
    TG1ProjAddAffine: G1ProjAddAffine<TG1, TG1Fp, TG1Affine>,
>(
    mut multi_points: Vec<Vec<TG1Affine>>,
) -> Vec<TG1> {
    multi_points
        .iter_mut()
        .for_each(|points| points.retain(|p| !p.is_infinity()));
    let total_num_points: usize = multi_points.iter().map(|p| p.len()).sum();
    let mut scratchpad = Vec::with_capacity(total_num_points);

    // Find the largest buckets, this will be the bottleneck for the number of iterations
    let mut max_bucket_length = 0;
    for points in multi_points.iter() {
        max_bucket_length = std::cmp::max(max_bucket_length, points.len());
    }

    // Compute the total number of "unit of work"
    // In the single batch addition case this is analogous to
    // the batch inversion threshold
    #[inline(always)]
    fn compute_threshold<TG1: G1, TG1Fp: G1Fp, TG1Affine: G1Affine<TG1, TG1Fp>>(
        points: &[Vec<TG1Affine>],
    ) -> usize {
        points
            .iter()
            .map(|p| {
                if p.len() % 2 == 0 {
                    p.len() / 2
                } else {
                    (p.len() - 1) / 2
                }
            })
            .sum()
    }

    let mut denominators = Vec::with_capacity(max_bucket_length);
    let mut total_amount_of_work = compute_threshold(&multi_points);

    let mut sums = vec![TG1::identity(); multi_points.len()];

    assert!(
        BATCH_INVERSE_THRESHOLD >= 2,
        "THRESHOLD cannot be below the number of points needed for group addition"
    );
    // TODO: total_amount_of_work does not seem to be changing performance that much
    while total_amount_of_work > BATCH_INVERSE_THRESHOLD {
        // For each point, we check if they are odd and pop off
        // one of the points
        for (points, sum) in multi_points.iter_mut().zip(sums.iter_mut()) {
            // Make the number of points even
            if points.len() % 2 != 0 {
                TG1ProjAddAffine::add_or_double_assign_affine(sum, &points.pop().unwrap());
            }
        }

        denominators.clear();

        // For each pair of points over all
        // vectors, we collect them and put them in the
        // inverse array
        for points in multi_points.iter_mut() {
            if points.len() < 2 {
                continue;
            }

            for pair in points.chunks_exact_mut(2) {
                denominators.push(choose_add_or_double(pair));
            }
        }

        batch_inverse_scratch_pad(&mut denominators, &mut scratchpad);

        let mut denominators_offset = 0;

        for points in multi_points.iter_mut() {
            if points.len() < 2 {
                continue;
            }

            for (i, inv) in (0..=points.len() - 2)
                .step_by(2)
                .zip(&denominators[denominators_offset..])
            {
                points[i / 2] = point_add_double(points[i], points[i + 1], inv);
            }

            let num_points = points.len() / 2;
            // The latter half of the vector is now unused,
            // all results are stored in the former half.
            points.truncate(num_points);
            denominators_offset += num_points
        }

        total_amount_of_work = compute_threshold(&multi_points);
    }

    for (sum, points) in sums.iter_mut().zip(multi_points) {
        for point in points {
            TG1ProjAddAffine::add_or_double_assign_affine(sum, &point);
        }
    }

    sums
}
