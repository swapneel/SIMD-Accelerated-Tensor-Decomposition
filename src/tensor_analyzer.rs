use crate::Tensor;

pub struct TensorProperties {
    pub shape: Vec<usize>,
}

pub fn analyze_tensor(tensor: &Tensor) -> TensorProperties {
    TensorProperties {
        shape: tensor.shape.clone(),
    }
}
