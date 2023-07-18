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
    let data: Array2<f64> = read_npy("data/data.npy").unwrap();
    println!("{}", data);

    let responsibilities: Array2<f64> = read_npy("data/responsibilities.npy").unwrap();
    println!("{}", responsibilities);

    let means: Array2<f64> = read_npy("data/means.npy").unwrap();
    println!("{}", means);

    let covs: Array3<f64> = read_npy("data/covs.npy").unwrap();
    println!("{}", covs);

    let weights: Array1<f64> = read_npy("data/weights.npy").unwrap();
    println!("{}", weights);

    let (means_computed, covs_computed, weights_computed) =
        maximize(data.view(), responsibilities.view());


    println!("{}", means_computed);

    assert!(means_computed.abs_diff_eq(&means, 1e-9));
    assert!(covs_computed.abs_diff_eq(&covs, 1e-9));
    assert!(weights_computed.abs_diff_eq(&weights, 1e-9));
}
