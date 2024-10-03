#[cfg(test)]
mod recover_tests {
    use kzg_bench::tests::recover::*;
    use rust_kzg_arkworks::kzg_proofs::LFFTSettings;
    use rust_kzg_arkworks::kzg_types::ArkFr as Fr;
    use rust_kzg_arkworks::utils::PolyData;

    #[test]
    fn recover_simple_() {
        recover_simple::<Fr, LFFTSettings, PolyData, PolyData>();
    }

    //Could be not working because of zero poly.
    #[test]
    fn recover_random_() {
        recover_random::<Fr, LFFTSettings, PolyData, PolyData>();
    }

    #[test]
    fn more_than_half_missing_() {
        more_than_half_missing::<Fr, LFFTSettings, PolyData, PolyData>();
    }
}
