use numpy::{IntoPyArray, PyReadonlyArray2, PyArray2, PyArray3, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;

fn maximize(data: ArrayView2<f64>, responsibilities: ArrayView2<f64>) -> (Array2<f64>, Array3<f64>, Array1<f64>) {

    let k = if let [_n, k] = responsibilities.shape() {
        k
    } else {panic!()};
    let d = if let [_n, d] = data.shape() {
        d
    } else {panic!()};

    let sum_responsibilities = responsibilities.sum_axis(Axis(0));

    let means = (&responsibilities.slice(s![.., .., NewAxis]) * &data.slice(s![.., NewAxis, ..]))
        .sum_axis(Axis(0))
        / sum_responsibilities.slice(s![.., NewAxis]);

    // n x k x d
    let adjusted = &data.slice(s![.., NewAxis, ..]) - &means.slice(s![NewAxis, .., ..]);

    let mut covs = Array3::<f64>::zeros((*k, *d, *d));

    (adjusted.axis_iter(Axis(1)), covs.axis_iter_mut(Axis(0)), responsibilities.axis_iter(Axis(1)))
        .into_par_iter()
        .for_each(|(x, mut cov, resp)| {
            let y = &x * &resp.slice(s![.., NewAxis]);
            cov += &x.t().dot(&y);
        });

    covs = &covs / &sum_responsibilities.slice(s![..,NewAxis,NewAxis]);


    let weights = &sum_responsibilities / sum_responsibilities.sum();


    (means, covs, weights)
}


#[pymodule]
fn gmm(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // example using immutable borrows producing a new array
    // fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    //     a * &x + &y
    // }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "maximize")]
    fn maximize_py<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        responsibilities: PyReadonlyArray2<f64>,
    ) -> (&'py PyArray2<f64>, &'py PyArray3<f64>, &'py PyArray1<f64> ){
        let data = data.as_array();
        let responsibilities = responsibilities.as_array();
        let (means, covs, weights) = maximize(data, responsibilities);
        (means.into_pyarray(py), covs.into_pyarray(py), weights.into_pyarray(py))
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_npy::read_npy;

    #[test]
    fn test_maximize() {
        let data: Array2<f64> = read_npy("./examples/data.npy").unwrap();
        let responsibilities: Array2<f64> = read_npy("./examples//responsibilities.npy").unwrap();
        let means: Array2<f64> = read_npy("./examples/means.npy").unwrap();

        let (means_computed, _, _) = maximize(data.view(), responsibilities.view());
        assert!(!means_computed.abs_diff_eq(&means, 1e-1));

    }
}
