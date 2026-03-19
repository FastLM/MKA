#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_route_mka_forward(
    torch::Tensor q,
    torch::Tensor l1,
    torch::Tensor l2,
    torch::Tensor l3,
    torch::Tensor lam,
    torch::Tensor wk,
    torch::Tensor wv);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fused_route_mka_forward_checked(
    torch::Tensor q,
    torch::Tensor l1,
    torch::Tensor l2,
    torch::Tensor l3,
    torch::Tensor lam,
    torch::Tensor wk,
    torch::Tensor wv) {
  CHECK_INPUT(q);
  CHECK_INPUT(l1);
  CHECK_INPUT(l2);
  CHECK_INPUT(l3);
  CHECK_INPUT(lam);
  CHECK_INPUT(wk);
  CHECK_INPUT(wv);
  return fused_route_mka_forward(q, l1, l2, l3, lam, wk, wv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_route_mka_forward_checked, "Fused Route-MKA forward (CUDA)");
}
