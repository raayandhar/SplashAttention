#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor Q_idx, torch::Tensor K_idx, float sm_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
