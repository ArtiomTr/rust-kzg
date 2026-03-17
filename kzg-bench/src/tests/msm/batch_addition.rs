use kzg::{
    msm::batch_addition::multi_batch_addition_binary_tree_stride, EcBackend, G1Affine,
    G1ProjAddAffine, G1,
};
use rand::{seq::SliceRandom, thread_rng, Rng};

fn expand_points_with_zeros<B: EcBackend>(
    mut points: Vec<B::G1Affine>,
    depth: usize,
    zero_insertion_chance: f64,
) -> Vec<B::G1Affine> {
    assert!((0.0..=1.0).contains(&zero_insertion_chance));

    let mut rng = rand::thread_rng();

    for _ in 0..depth {
        let mut expanded = Vec::with_capacity(points.len() * 3 + 1);

        if rng.gen_bool(zero_insertion_chance) {
            expanded.push(B::G1Affine::zero());
        }

        for p in points {
            let q = B::G1::rand();
            let p_minus_q = p.to_proj().sub(&q);

            expanded.push(B::G1Affine::into_affine(&p_minus_q));
            expanded.push(B::G1Affine::into_affine(&q));

            if rng.gen_bool(zero_insertion_chance) {
                expanded.push(B::G1Affine::zero());
            }
        }

        points = expanded;
    }

    points
}

/// This test checks if addition works correctly, when only zeros are provided
/// as input.
pub fn test_all_infinity_points<B: EcBackend>() {
    let points = vec![
        vec![B::G1Affine::zero(); 5],
        vec![B::G1Affine::zero(); 18],
        vec![B::G1Affine::zero(); 32],
    ];

    let result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            points,
        );

    assert_eq!(result, vec![B::G1::zero(); 3]);
}

/// This test case puts point and its inverse side-by-side, so that after first
/// batch addition iteration we already have all zeros as a results.
pub fn test_opposite_pair_reduction<B: EcBackend>() {
    let mut points = Vec::new();

    // At least 34 points (17 additions) are required, otherwise batch reduction
    //   won't trigger, as one-by-one addition is cheaper.
    for _ in 0..17 {
        let p = B::G1Affine::into_affine(&B::G1::rand());
        points.push(p.neg());
        points.push(p);
    }

    let result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(result, vec![B::G1::zero()]);
}

/// This test is similar to `test_opposite_pair_reduction` - generating N random
/// points, then calculating their inverses. After that, we shuffle all
/// generated points and their inverses. As a result, we will probably get lots
/// of intermediatery zeros, on various batch addition iteration stages.
pub fn test_shuffled_opposite_pair_reduction<B: EcBackend>() {
    let mut points = Vec::new();

    // We need at least 17 additions to trigger batch addition algorithm, but we
    // generate much more pairs, just to increase chance of encountering zeros
    // in intermediatery operations.
    for _ in 0..64 {
        let p = B::G1Affine::into_affine(&B::G1::rand());
        points.push(p.neg());
        points.push(p);
    }

    points.shuffle(&mut thread_rng());

    let result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(result, vec![B::G1::zero()]);
}

/// This test forces sudden appearance of a layer with only zero points. Done
/// with a `expand_points_with_zeros` helper method, that iteratively expands
/// point set, generating random points, that don't look like zeros.
pub fn test_full_infinity_point_layer<B: EcBackend>() {
    // At least 17 additions are required to trigger batch.
    let points = vec![B::G1Affine::zero(); 34];

    let points = expand_points_with_zeros::<B>(points, 3, 0.0);

    let received_result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(received_result, vec![B::G1::zero()]);
}

/// This test forces zero point occurence on some itermediatery batch addition
/// level. This is done via `expand_points_with_zeros` helper method, that
/// iteratively expand point set, generating random point pairs, sum of which
/// results in needed point set.
pub fn test_one_layer_infinity_point_reduction<B: EcBackend>() {
    let mut points = Vec::new();

    // We need at least 17 additions to trigger batch addition algorithm.
    for _ in 0..34 {
        points.push(B::G1Affine::into_affine(&B::G1::rand()));
    }

    // Put some zero points into our array
    points[2] = B::G1Affine::zero();
    points[15] = B::G1Affine::zero();
    points[16] = B::G1Affine::zero();
    points[33] = B::G1Affine::zero();

    // Calculate expected result
    let mut expected_result = B::G1::zero();
    for p in &points {
        B::G1ProjAddAffine::add_assign_affine(&mut expected_result, p);
    }

    // Expand points, so that initial array won't contain any zeros
    let points = expand_points_with_zeros::<B>(points, 3, 0.0);

    let received_result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(received_result.len(), 1);
    assert_eq!(
        B::G1Affine::into_affine(&received_result[0]),
        B::G1Affine::into_affine(&expected_result)
    );
}

/// This test ensures that zeros appear on multiple layers during batch addition
/// iteration. This also constructs unbalanced tree, so that some additions
/// won't have ideal pairs.
pub fn test_multi_layer_infinity_point_reduction<B: EcBackend>() {
    let mut points = Vec::new();

    // We need at least 17 additions to trigger batch addition algorithm.
    for _ in 0..34 {
        points.push(B::G1Affine::into_affine(&B::G1::rand()));
    }

    // Put some zero points into our array
    points[5] = B::G1Affine::zero();
    points[9] = B::G1Affine::zero();
    points[23] = B::G1Affine::zero();
    points[31] = B::G1Affine::zero();

    // Calculate expected result
    let mut expected_result = B::G1::zero();
    for p in &points {
        B::G1ProjAddAffine::add_assign_affine(&mut expected_result, p);
    }

    // Expand points, so that initial array won't contain any zeros
    let points = expand_points_with_zeros::<B>(points, 3, 0.5);

    let received_result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(received_result.len(), 1);
    assert_eq!(
        B::G1Affine::into_affine(&received_result[0]),
        B::G1Affine::into_affine(&expected_result)
    );
}

/// This test ensures that zeros appear on every layer of batch addition
/// iteration. This is a stress test for batch addition reduction.
pub fn test_all_layer_infinity_point_reduction<B: EcBackend>() {
    let mut points = Vec::new();

    // We need at least 17 additions to trigger batch addition algorithm.
    for _ in 0..34 {
        points.push(B::G1Affine::into_affine(&B::G1::rand()));
    }

    // Calculate expected result
    let mut expected_result = B::G1::zero();
    for p in &points {
        B::G1ProjAddAffine::add_assign_affine(&mut expected_result, p);
    }

    // Expand points, so that initial array won't contain any zeros
    let points = expand_points_with_zeros::<B>(points, 4, 1.0);

    let received_result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(received_result.len(), 1);
    assert_eq!(
        B::G1Affine::into_affine(&received_result[0]),
        B::G1Affine::into_affine(&expected_result)
    );
}

/// This test fills point array with random values. No special cases - pure
/// batch addition.
pub fn test_random_point_reduction<B: EcBackend>() {
    let mut points = Vec::new();

    // We need at least 17 additions to trigger batch addition algorithm.
    for _ in 0..34 {
        points.push(B::G1Affine::into_affine(&B::G1::rand()));
    }

    // Calculate expected result
    let mut expected_result = B::G1::zero();
    for p in &points {
        B::G1ProjAddAffine::add_assign_affine(&mut expected_result, p);
    }

    let received_result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(received_result.len(), 1);
    assert_eq!(
        B::G1Affine::into_affine(&received_result[0]),
        B::G1Affine::into_affine(&expected_result)
    );
}

/// This test creates an array of numbers, with amount of items that is a
/// [Mersenne number]. This, in turn, makes tree as unbalanced as possible,
/// requiring performing additional sum operation on each iteration of tree.
///
/// Quick note: Mersenne number is just a number, which in binary would look
/// like all ones (1=0b1, 3=0b11, 7=0b111, etc.). This way, when we recursively
/// subtract 1 and divide this number by 2, we will get an odd number each time.
/// This is the worst case for batch addition, as batch addition amortizes
/// inverses only for even number of points, as it operates on pairs, and after
/// each iteration it reduces amount of points by a factor of 2.
///
/// [Mersenne number]: https://mathworld.wolfram.com/MersenneNumber.html
pub fn test_random_point_reduction_unbalanced_tree<B: EcBackend>() {
    // Tunable parameter. Should not be less than 6, as with 5 the number of
    // points generated will be 31, and this amount of points requires only 15
    // additions for first iteration, what is less than threshold, so batch
    // addition won't kick in.
    let n = 6;
    let numpoints = 1usize << (n - 1); // Generate Mersenne number.

    // Generate desired amount of points.
    let mut points = Vec::with_capacity(numpoints);
    for _ in 0..numpoints {
        points.push(B::G1Affine::into_affine(&B::G1::rand()));
    }

    // Calculate expected result
    let mut expected_result = B::G1::zero();
    for p in &points {
        B::G1ProjAddAffine::add_assign_affine(&mut expected_result, p);
    }

    let received_result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            vec![points],
        );

    assert_eq!(received_result.len(), 1);
    assert_eq!(
        B::G1Affine::into_affine(&received_result[0]),
        B::G1Affine::into_affine(&expected_result)
    );
}

/// Test case, that generates multiple sets of random points.
pub fn test_random_multi_point_reduction<B: EcBackend>() {
    let numbatches = 34;
    let mut points = Vec::with_capacity(numbatches);
    let mut expected_results = Vec::with_capacity(numbatches);

    let mut rng = thread_rng();
    for _ in 0..numbatches {
        let numpoints = rng.gen_range(0..=1024);

        let mut curr_points = Vec::with_capacity(numpoints);

        let mut res = B::G1::zero();
        for _ in 0..numpoints {
            let point = B::G1Affine::into_affine(&B::G1::rand());
            B::G1ProjAddAffine::add_assign_affine(&mut res, &point);
            curr_points.push(point);
        }
        expected_results.push(B::G1Affine::into_affine(&res));

        points.push(curr_points);
    }

    let received_result =
        multi_batch_addition_binary_tree_stride::<B::G1, B::G1Fp, B::G1Affine, B::G1ProjAddAffine>(
            points,
        );

    let received_results = received_result
        .iter()
        .map(B::G1Affine::into_affine)
        .collect::<Vec<_>>();
    assert_eq!(expected_results, received_results);
}

#[macro_export]
macro_rules! instantiate_batch_addition_tests {
    ($backend: ty) => {
        #[test]
        fn test_all_infinity_points() {
            kzg_bench::tests::msm::batch_addition::test_all_infinity_points::<$backend>();
        }

        #[test]
        fn test_opposite_pair_reduction() {
            kzg_bench::tests::msm::batch_addition::test_opposite_pair_reduction::<$backend>();
        }

        #[test]
        fn test_shuffled_opposite_pair_reduction() {
            kzg_bench::tests::msm::batch_addition::test_shuffled_opposite_pair_reduction::<$backend>();
        }

        #[test]
        fn test_full_infinity_point_layer() {
            kzg_bench::tests::msm::batch_addition::test_full_infinity_point_layer::<$backend>();
        }

        #[test]
        fn test_one_layer_infinity_point_reduction() {
            kzg_bench::tests::msm::batch_addition::test_one_layer_infinity_point_reduction::<$backend>();
        }

        #[test]
        fn test_multi_layer_infinity_point_reduction() {
            kzg_bench::tests::msm::batch_addition::test_multi_layer_infinity_point_reduction::<$backend>();
        }

        #[test]
        fn test_all_layer_infinity_point_reduction() {
            kzg_bench::tests::msm::batch_addition::test_all_layer_infinity_point_reduction::<$backend>();
        }

        #[test]
        fn test_random_point_reduction() {
            kzg_bench::tests::msm::batch_addition::test_random_point_reduction::<$backend>();
        }

        #[test]
        fn test_random_point_reduction_unbalanced_tree() {
            kzg_bench::tests::msm::batch_addition::test_random_point_reduction_unbalanced_tree::<$backend>();
        }

        #[test]
        fn test_random_multi_point_reduction() {
            kzg_bench::tests::msm::batch_addition::test_random_multi_point_reduction::<$backend>();
        }
    };
}
