#[cfg(test)]
mod tests {
    use kzg_bench::instantiate_batch_addition_tests;
    use rust_kzg_constantine::eip_7594::CtBackend;

    instantiate_batch_addition_tests!(CtBackend);
}
