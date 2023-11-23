#[cfg(test)]
mod tests {
    use std::iter::zip;
    use std::path::PathBuf;
    use std::{fs::File, io::Read};

    use blst::{blst_fp, blst_p1_affine, BLST_ERROR};
    use rust_kzg_blst::bgmw::{self, get_bgmw_table_size};

    fn get_fixture_path(fixture: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture)
            .join("data.txt")
    }

    #[test]
    fn bgmw_precomputation_test() {
        struct Fixture {
            name: String,
            message: String,
        }

        let fixtures = [
            Fixture {
                name: "valid_random_points_bgmw_precomputation".to_string(),
                message: "Precomputing small random point set".to_string(),
            },
            Fixture {
                name: "valid_p1_generator_bgmw_precomputation".to_string(),
                message: "Precomputing random point set".to_string(),
            },
            Fixture {
                name: "valid_trusted_setup_bgmw_precomputation".to_string(),
                message: "Precomputing trusted setup points".to_string(),
            },
        ];

        for fixture in fixtures {
            let file_path = get_fixture_path(&fixture.name);
            let mut file = File::open(file_path.clone()).unwrap();

            let mut contents = String::new();

            file.read_to_string(&mut contents).unwrap();

            let mut lines = contents.lines();

            let point_count = lines.next().unwrap().parse::<usize>().unwrap();
            let table_size = lines.next().unwrap().parse::<usize>().unwrap();

            assert_eq!(table_size, get_bgmw_table_size(point_count));

            let points = (0..point_count)
                .map(|_| {
                    let bytes = hex::decode(lines.next().unwrap()).unwrap();

                    let mut output = blst_p1_affine::default();

                    let code = unsafe { blst::blst_p1_uncompress(&mut output, bytes.as_ptr()) };

                    assert_eq!(code, BLST_ERROR::BLST_SUCCESS);

                    output
                })
                .collect::<Vec<blst_p1_affine>>();

            let expected_table = (0..table_size)
                .map(|_| {
                    let bytes = hex::decode(lines.next().unwrap()).unwrap();

                    let mut output = blst_p1_affine::default();

                    let code = unsafe { blst::blst_p1_uncompress(&mut output, bytes.as_ptr()) };

                    assert_eq!(code, BLST_ERROR::BLST_SUCCESS);

                    output
                })
                .collect::<Vec<blst_p1_affine>>();

            let mut received_table =
                vec![blst_p1_affine::default(); get_bgmw_table_size(point_count)];

            bgmw::init_pippenger_bgmw(&mut received_table, points.as_slice());

            for (i, (expected, received)) in zip(expected_table, received_table).enumerate() {
                assert_eq!(
                    expected,
                    received,
                    "{}, points at index {} are not equal, fixture = {}",
                    fixture.message,
                    i,
                    file_path.as_os_str().to_str().unwrap()
                );
            }
        }
    }

    #[test]
    fn single_scalar_multiplication_test() {
        let scalar: usize = 2;
        let mut test: blst_p1_affine = blst_p1_affine {
            x: blst_fp {
                l: [
                    6046496802367715900,
                    4512703842675942905,
                    5557647857818872160,
                    11911007586355426777,
                    2789226406901363231,
                    2402832991291269,
                ],
            },
            y: blst_fp {
                l: [
                    8075247918781118784,
                    15723127573743364860,
                    13289805640942397317,
                    12593984073093990549,
                    2724610382811436832,
                    447576566110657301,
                ],
            },
        };

        bgmw::single_scalar_multiplication(scalar, &mut test);

        let truth: blst_p1_affine = blst_p1_affine {
            x: blst_fp {
                l: [
                    829396425412873931,
                    4630191540216922912,
                    355385700763152750,
                    14285431475212315696,
                    805409597614920011,
                    576608384544293863,
                ],
            },
            y: blst_fp {
                l: [
                    16917084574279409168,
                    1285895338278983897,
                    8416868578363461231,
                    16052080159904405683,
                    17022798986550011276,
                    1782778860764068411,
                ],
            },
        };

        assert_eq!(test, truth)
    }
}
