#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

#define FASTMKA_MAX_D 256
#define FASTMKA_TILE_K 64

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
__global__ void fastmka_attn_kernel(
    const scalar_t* __restrict__ q,  // [B,H,Tq,D]
    const scalar_t* __restrict__ k,  // [B,H,Tk,D]
    const scalar_t* __restrict__ v,  // [B,H,Tk,D]
    scalar_t* __restrict__ out,      // [B,H,Tq,D]
    int B,
    int H,
    int Tq,
    int Tk,
    int D,
    bool causal) {
  const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = B * H * Tq;
  if (row_id >= total_rows) return;

  const int tq = row_id % Tq;
  const int bh = row_id / Tq;
  const int h = bh % H;
  const int b = bh / H;

  const int q_base = (((b * H + h) * Tq + tq) * D);
  const int kv_base = ((b * H + h) * Tk * D);
  const int q_offset = Tk - Tq;

  float m = -std::numeric_limits<float>::infinity();
  float l = 0.0f;

  float acc[FASTMKA_MAX_D];
  for (int d = 0; d < D; ++d) acc[d] = 0.0f;

  for (int tk0 = 0; tk0 < Tk; tk0 += FASTMKA_TILE_K) {
    const int tk1 = min(tk0 + FASTMKA_TILE_K, Tk);
    for (int tk = tk0; tk < tk1; ++tk) {
      if (causal && tk > (tq + q_offset)) continue;

      float score = 0.0f;
      const int k_off = kv_base + tk * D;
      for (int d = 0; d < D; ++d) {
        score += to_float(q[q_base + d]) * to_float(k[k_off + d]);
      }
      score /= sqrtf(static_cast<float>(D));

      const float m_new = fmaxf(m, score);
      const float alpha = expf(m - m_new);
      const float beta = expf(score - m_new);

      const int v_off = kv_base + tk * D;
      for (int d = 0; d < D; ++d) {
        acc[d] = acc[d] * alpha + beta * to_float(v[v_off + d]);
      }
      l = l * alpha + beta;
      m = m_new;
    }
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int d = 0; d < D; ++d) {
    out[q_base + d] = static_cast<scalar_t>(acc[d] * inv_l);
  }
}

torch::Tensor fastmka_attn_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA");
  TORCH_CHECK(v.is_cuda(), "v must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1), "B/H mismatch");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have same shape");
  TORCH_CHECK(q.size(3) == k.size(3), "head dim mismatch");

  const auto B = q.size(0);
  const auto H = q.size(1);
  const auto Tq = q.size(2);
  const auto D = q.size(3);
  const auto Tk = k.size(2);
  TORCH_CHECK(D <= FASTMKA_MAX_D, "head_dim exceeds FASTMKA_MAX_D=256");
  auto out = torch::zeros_like(q);

  const int threads = 128;
  const int total_rows = B * H * Tq;
  const int blocks = (total_rows + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "fastmka_attn_cuda", [&] {
    fastmka_attn_kernel<scalar_t><<<blocks, threads>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        B,
        H,
        Tq,
        Tk,
        D,
        causal);
  });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fastmka_attn_cuda launch failed: ", cudaGetErrorString(err));
  return out;
}
