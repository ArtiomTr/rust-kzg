#[cfg(test)]
mod tests {
    use kzg_bench::instantiate_batch_addition_tests;
    use rust_kzg_blst::eip_7594::BlstBackend;

    instantiate_batch_addition_tests!(BlstBackend);
}
