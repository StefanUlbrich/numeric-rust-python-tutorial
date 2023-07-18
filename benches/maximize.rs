use criterion::{black_box, criterion_group, criterion_main, Criterion};
// use itertools::izip;
// use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_npy::read_npy;

struct Input {
    data: Array2<f64>,
    responsibilities: Array2<f64>,
}

fn data_fixture() -> Input {
    // Note: we need to explicitly state the type here ..
    let data: Array2<f64> = read_npy("data/data.npy").unwrap();
    let responsibilities: Array2<f64> = read_npy("data/responsibilities.npy").unwrap();

    Input {
        data,
        responsibilities,
    }
}

fn with_loops(input: Input) {}

fn with_loops_bench(c: &mut Criterion) {
    c.bench_function("with_loops", |b| b.iter(|| with_loops(data_fixture())));
}

criterion_group!(benches, with_loops_bench);
criterion_main!(benches);
