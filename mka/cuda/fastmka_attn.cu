#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

#define FASTMKA_MAX_D 256
#define FASTMKA_THREADS 128
#define FASTMKA_WARP_SIZE 32
#define FASTMKA_MAX_WARPS (FASTMKA_THREADS / FASTMKA_WARP_SIZE)

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float x) {
  return static_cast<scalar_t>(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = FASTMKA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float warp_sums[FASTMKA_MAX_WARPS];
  const int lane = threadIdx.x & (FASTMKA_WARP_SIZE - 1);
  const int warp_id = threadIdx.x / FASTMKA_WARP_SIZE;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    warp_sums[warp_id] = val;
  }
  __syncthreads();

  float out = 0.0f;
  if (warp_id == 0) {
    const int nwarps = (blockDim.x + FASTMKA_WARP_SIZE - 1) / FASTMKA_WARP_SIZE;
    out = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    out = warp_reduce_sum(out);
  }
  return out;
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
  const int row_id = blockIdx.x;
  const int total_rows = B * H * Tq;
  if (row_id >= total_rows) return;

  const int tq = row_id % Tq;
  const int bh = row_id / Tq;
  const int h = bh % H;
  const int b = bh / H;

  const int q_base = (((b * H + h) * Tq + tq) * D);
  const int kv_base = ((b * H + h) * Tk * D);
  const int q_offset = Tk - Tq;

  __shared__ float q_sh[FASTMKA_MAX_D];
  __shared__ float acc_sh[FASTMKA_MAX_D];
  __shared__ float alpha_sh;
  __shared__ float beta_sh;
  __shared__ float m_sh;
  __shared__ float l_sh;

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_sh[d] = to_float(q[q_base + d]);
    acc_sh[d] = 0.0f;
  }
  if (threadIdx.x == 0) {
    m_sh = -std::numeric_limits<float>::infinity();
    l_sh = 0.0f;
  }
  __syncthreads();

  float m = -std::numeric_limits<float>::infinity();
  float l = 0.0f;

  const float scale = rsqrtf(static_cast<float>(D));
  const int tk_limit = causal ? min(Tk - 1, tq + q_offset) : (Tk - 1);
  for (int tk = 0; tk <= tk_limit; ++tk) {
    const int k_off = kv_base + tk * D;

    float thread_dot = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      thread_dot += q_sh[d] * to_float(k[k_off + d]);
    }
    float block_dot = block_reduce_sum(thread_dot);

    if (threadIdx.x == 0) {
      const float score = block_dot * scale;
      const float m_new = fmaxf(m_sh, score);
      alpha_sh = expf(m_sh - m_new);
      beta_sh = expf(score - m_new);
      l_sh = l_sh * alpha_sh + beta_sh;
      m_sh = m_new;
    }
    __syncthreads();

    const int v_off = kv_base + tk * D;
    const float alpha = alpha_sh;
    const float beta = beta_sh;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      acc_sh[d] = acc_sh[d] * alpha + beta * to_float(v[v_off + d]);
    }
    __syncthreads();
  }

  m = m_sh;
  l = l_sh;
  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    out[q_base + d] = from_float<scalar_t>(acc_sh[d] * inv_l);
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

  const int threads = (D <= 64) ? 64 : FASTMKA_THREADS;
  const int blocks = static_cast<int>(B * H * Tq);

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
