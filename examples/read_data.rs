use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_npy::read_npy;

fn maximize(
    data: ArrayView2<f64>,
    responsibilities: ArrayView2<f64>,
) -> (Array2<f64>, Array3<f64>, Array1<f64>) {
    (
        Array2::<f64>::zeros((0, 0)),
        Array3::<f64>::zeros((0, 0, 0)),
        Array1::<f64>::zeros(0),
    )
}

fn main() {
    let data: Array2<f64> = read_npy("examples/data/data.npy").unwrap();
    println!("{}", data);

    let responsibilities: Array2<f64> = read_npy("examples/data/responsibilities.npy").unwrap();
    println!("{}", responsibilities);

    let means: Array2<f64> = read_npy("examples/data/means.npy").unwrap();
    println!("{}", means);

    let (means_computed, _, _) = maximize(data.view(), responsibilities.view());

    println!("{}", means_computed);

    assert!(means_computed.abs_diff_eq(&means, 1e-9));
}
