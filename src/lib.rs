mod hardware_detection;
mod tensor_analyzer;
mod simd_optimizations;
mod decomposition_algorithms;
mod utils;

pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }

}

pub fn decompose(tensor: &Tensor) {
    let capabilities = hardware_detection::detect_simd_capabilities();
    let tensor_properties = tensor_analyzer::analyze_tensor(tensor);
    match capabilities {
        hardware_detection::SimdCapabilities::Avx512 => {
            // todo
        },
        hardware_detection::SimdCapabilities::Avx2 => {
            // todo
        },
        _ => {
        }
    }
}
