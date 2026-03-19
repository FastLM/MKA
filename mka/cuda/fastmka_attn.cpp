#include <torch/extension.h>

torch::Tensor fastmka_attn_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor fastmka_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
  return fastmka_attn_cuda(q, k, v, causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fastmka_attn", &fastmka_attn, "FastMKA attention CUDA");
}
