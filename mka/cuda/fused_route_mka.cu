#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

/*
  Fused FastMKA forward: for each query position, form fused memory from
  (l1,l2,l3) and router weights lam, project to K/V with wk/wv (nn.Linear
  weights [D,D] row-major: out[o] = sum_i w[o,i]*x[i]), then causal softmax
  attention with the same online stable softmax as fastmka_attn.
*/

#define FASTMKA_ROUTE_MAX_DH 256

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
__global__ void fused_route_mka_forward_kernel(
    const scalar_t* __restrict__ q,   // [B,H,T,Dh]
    const scalar_t* __restrict__ l1,  // [B,T,D]
    const scalar_t* __restrict__ l2,  // [B,T,D]
    const scalar_t* __restrict__ l3,  // [B,T,D] (unused when L==2)
    const scalar_t* __restrict__ lam, // [B,T,L]  L in {2,3}
    const scalar_t* __restrict__ wk,  // [D,D]  weight[out,i] at wk[out*D+i]
    const scalar_t* __restrict__ wv,  // [D,D]
    scalar_t* __restrict__ out,       // [B,H,T,Dh]
    int B,
    int H,
    int T,
    int D,
    int Dh,
    int L) {
  const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = B * H * T;
  if (row_id >= total_rows) return;

  const int tq = row_id % T;
  const int bh = row_id / T;
  const int h = bh % H;
  const int b = bh / H;

  const float scale = 1.0f / sqrtf(static_cast<float>(Dh));

  float m = -std::numeric_limits<float>::infinity();
  float lsum = 0.0f;

  float acc[FASTMKA_ROUTE_MAX_DH];
  float k_vec[FASTMKA_ROUTE_MAX_DH];
  float v_vec[FASTMKA_ROUTE_MAX_DH];
  for (int j = 0; j < Dh; ++j) acc[j] = 0.0f;

  const int q_base = ((b * H + h) * T + tq) * Dh;
  const int out_row_k = h * Dh;

  for (int tk = 0; tk <= tq; ++tk) {
    const int lam_off = (b * T + tk) * L;
    const int mem_off = (b * T + tk) * D;

    for (int j = 0; j < Dh; ++j) {
      const int w_row = (out_row_k + j) * D;
      float ksum = 0.0f;
      float vsum = 0.0f;
      for (int i = 0; i < D; ++i) {
        float fused_i = to_float(lam[lam_off + 0]) * to_float(l1[mem_off + i]);
        fused_i += to_float(lam[lam_off + 1]) * to_float(l2[mem_off + i]);
        if (L > 2) {
          fused_i += to_float(lam[lam_off + 2]) * to_float(l3[mem_off + i]);
        }
        ksum += to_float(wk[w_row + i]) * fused_i;
        vsum += to_float(wv[w_row + i]) * fused_i;
      }
      k_vec[j] = ksum;
      v_vec[j] = vsum;
    }

    float score = 0.0f;
    for (int j = 0; j < Dh; ++j) {
      score += to_float(q[q_base + j]) * k_vec[j];
    }
    score *= scale;

    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float beta = expf(score - m_new);

    for (int j = 0; j < Dh; ++j) {
      acc[j] = acc[j] * alpha + beta * v_vec[j];
    }
    lsum = lsum * alpha + beta;
    m = m_new;
  }

  const float inv_l = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
  for (int j = 0; j < Dh; ++j) {
    out[q_base + j] = static_cast<scalar_t>(acc[j] * inv_l);
  }
}

std::vector<torch::Tensor> fused_route_mka_forward(
    torch::Tensor q,
    torch::Tensor l1,
    torch::Tensor l2,
    torch::Tensor l3,
    torch::Tensor lam,
    torch::Tensor wk,
    torch::Tensor wv) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(q.dim() == 4, "q must be [B,H,T,Dh]");
  TORCH_CHECK(l1.dim() == 3 && l2.dim() == 3 && l3.dim() == 3, "l1/l2/l3 must be [B,T,D]");
  TORCH_CHECK(lam.dim() == 3, "lam must be [B,T,L]");
  TORCH_CHECK(wk.dim() == 2 && wv.dim() == 2, "wk/wv must be [D,D]");

  const int B = static_cast<int>(q.size(0));
  const int H = static_cast<int>(q.size(1));
  const int T = static_cast<int>(q.size(2));
  const int Dh = static_cast<int>(q.size(3));
  const int D = static_cast<int>(l1.size(2));
  const int L = static_cast<int>(lam.size(2));

  TORCH_CHECK(l1.sizes() == l2.sizes() && l1.sizes() == l3.sizes(), "l1/l2/l3 shapes must match");
  TORCH_CHECK(l1.size(0) == B && l1.size(1) == T, "l1 batch/seq must match q");
  TORCH_CHECK(lam.size(0) == B && lam.size(1) == T, "lam batch/seq must match q");
  TORCH_CHECK((L == 2) || (L == 3), "lam last dim must be 2 or 3");
  TORCH_CHECK(wk.size(0) == D && wk.size(1) == D, "wk must be [D,D] with D = hidden_size");
  TORCH_CHECK(wv.sizes() == wk.sizes(), "wv must match wk shape");
  TORCH_CHECK(H * Dh == D, "expected H*Dh == D (hidden_size)");
  TORCH_CHECK(Dh <= FASTMKA_ROUTE_MAX_DH, "Dh exceeds FASTMKA_ROUTE_MAX_DH=", FASTMKA_ROUTE_MAX_DH);

  auto out = torch::empty_like(q);

  const int total_rows = B * H * T;
  const int threads = 128;
  const int blocks = (total_rows + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "fused_route_mka_forward", [&] {
    fused_route_mka_forward_kernel<scalar_t><<<blocks, threads>>>(
        q.data_ptr<scalar_t>(),
        l1.data_ptr<scalar_t>(),
        l2.data_ptr<scalar_t>(),
        l3.data_ptr<scalar_t>(),
        lam.data_ptr<scalar_t>(),
        wk.data_ptr<scalar_t>(),
        wv.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        B,
        H,
        T,
        D,
        Dh,
        L);
  });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fused_route_mka_forward launch failed: ", cudaGetErrorString(err));
  return {out};
}
