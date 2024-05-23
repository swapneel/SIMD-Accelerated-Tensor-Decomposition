use crate::Tensor;
use ndarray::{Array, ArrayD, IxDyn};
use ndarray_linalg::{self, Scalar};

pub fn cp_decomposition(tensor: &Tensor, rank: usize) -> Vec<ArrayD<f32>> {
    let data = Array::from_shape_vec(IxDyn(&tensor.shape), tensor.data.clone()).unwrap();
    let factors = cp_decompose(&data, rank);
    factors
}

pub fn tucker_decomposition(tensor: &Tensor, ranks: &[usize]) -> (ArrayD<f32>, Vec<ArrayD<f32>>) {
    let data = Array::from_shape_vec(IxDyn(&tensor.shape), tensor.data.clone()).unwrap();
    let (core, factors) = tucker_decompose(&data, ranks);
    (core, factors)
}

fn cp_decompose(tensor: &ArrayD<f32>, rank: usize) -> Vec<ArrayD<f32>> {
    let shape = tensor.shape().to_vec();
    let mut factors = vec![];

    for &dim in &shape {
        let factor = Array::random((dim, rank), |x| x);
        factors.push(factor);
    }

    for _ in 0..100 {
        for mode in 0..shape.len() {
            let mut kr_prod = kronecker_product(&factors, mode);
            let unfold = unfold_tensor(tensor, mode);
            let new_factor = pseudo_inverse(&kr_prod).dot(&unfold);
            factors[mode] = new_factor;
        }
    }

    factors
}

fn tucker_decompose(tensor: &ArrayD<f32>, ranks: &[usize]) -> (ArrayD<f32>, Vec<ArrayD<f32>>) {
    let shape = tensor.shape().to_vec();
    let mut factors = vec![];

    for (&dim, &rank) in shape.iter().zip(ranks) {
        let factor = Array::random((dim, rank), |x| x);
        factors.push(factor);
    }

    // Alternating Least Squares for Tucker decomposition
    let mut core = tensor.clone();
    for _ in 0..100 {
        for mode in 0..shape.len() {
            let mut prod = core.clone();
            for (i, factor) in factors.iter().enumerate() {
                if i != mode {
                    prod = tensor_contract(&prod, &factor.t(), i);
                }
            }
            let unfold = unfold_tensor(tensor, mode);
            let new_factor = pseudo_inverse(&unfold).dot(&prod);
            factors[mode] = new_factor;
        }
    }

    (core, factors)
}

fn unfold_tensor(tensor: &ArrayD<f32>, mode: usize) -> ArrayD<f32> {
    let mut shape = tensor.shape().to_vec();
    let mut new_shape = vec![shape[mode]];
    shape.remove(mode);
    new_shape.extend(shape.iter());

    let mut unfolded = Array::zeros(IxDyn(&new_shape));
    for (i, slice) in tensor.lanes(Axis(mode)).into_iter().enumerate() {
        unfolded.index_axis_mut(Axis(0), i).assign(&slice.to_owned().into_shape(IxDyn(&shape)).unwrap());
    }

    unfolded
}

fn kronecker_product(factors: &[ArrayD<f32>], skip_mode: usize) -> ArrayD<f32> {
    let mut result = Array::ones(IxDyn(&[1]));
    for (i, factor) in factors.iter().enumerate() {
        if i != skip_mode {
            result = result.kronecker(factor);
        }
    }
    result
}

fn pseudo_inverse(matrix: &ArrayD<f32>) -> ArrayD<f32> {
    let svd = matrix.svd(true, true).unwrap();
    let (u, s, vt) = svd.unwrap();

    let s_inv = s.mapv(|x| if x > 1e-10 { 1.0 / x } else { 0.0 });
    let s_inv_diag = Array::from_diag(&s_inv);

    let pseudo_inv = vt.t().dot(&s_inv_diag).dot(&u.t());
    pseudo_inv
}

fn tensor_contract(tensor: &ArrayD<f32>, factor: &ArrayD<f32>, mode: usize) -> ArrayD<f32> {
    let mut shape = tensor.shape().to_vec();
    shape[mode] = factor.shape()[1];

    let mut result = Array::zeros(IxDyn(&shape));
    for (i, slice) in tensor.lanes(Axis(mode)).into_iter().enumerate() {
        result.index_axis_mut(Axis(mode), i).assign(&factor.dot(&slice));
    }

    result
}