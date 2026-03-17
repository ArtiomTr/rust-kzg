#[cfg(test)]
mod tests {
    use kzg_bench::instantiate_batch_addition_tests;
    use rust_kzg_arkworks3::eip_7594::ArkBackend;

    instantiate_batch_addition_tests!(ArkBackend);
}
