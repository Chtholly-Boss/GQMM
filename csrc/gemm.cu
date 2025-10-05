#include "binding.h"
#include "dequant.cuh"
#include "helper.cuh"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>

/// ! only support 16x8x32 for now, just 1 warp
template <int kCtaTileM, int kCtaTileN, int kCtaTileK>
static __global__ void k_gemm_bf16_mxfp4(nv_bfloat16 *A, uint32_t *W,
                                         uint8_t *scales, nv_bfloat16 *out,
                                         int m, int n, int k) {
  __shared__ nv_bfloat16 sA[kCtaTileM * kCtaTileK];
  __shared__ uint32_t
      sW[kCtaTileN * kCtaTileK / 8]; // 8 mxfp4 values per uint32_t

  auto const tid = threadIdx.x;

  for (int i = tid; i < kCtaTileM * kCtaTileK; i += blockDim.x) {
    sA[i] = A[i];
  }
  for (int i = tid; i < kCtaTileN * kCtaTileK / 8; i += blockDim.x) {
    sW[i] = W[i];
  }
  __syncthreads();

  auto thr_idx_mn = tid / 4;
  auto thr_idx_k = tid % 4;

  uint32_t a_frags[2][4]; // 2 rows of 8 contiguous bf16

  v_mov<uint4>(&a_frags[0], &sA[thr_idx_mn * kCtaTileK + thr_idx_k * 8]);
  v_mov<uint4>(&a_frags[1], &sA[(thr_idx_mn + 8) * kCtaTileK + thr_idx_k * 8]);

  uint32_t qb_frags; // 8 contiguous mxfp4
  qb_frags = sW[thr_idx_mn * kCtaTileK / 8 + thr_idx_k];

  uint32_t scale = scales[thr_idx_mn];

  // dequantization
  uint32_t b_frags[4];
  auto b_frags_bf16_ptr = reinterpret_cast<nv_bfloat16 *>(&b_frags);
  for (int i = 0; i < 8; i++) {
    auto mxfp4_val = (qb_frags >> (i * 4)) & 0xF;
    b_frags_bf16_ptr[i] = dequant_mxfp4_with_scale(mxfp4_val, scale);
  }

  float c_frags[2][2] = {
      {0.0f, 0.0f},
      {0.0f, 0.0f},
  };
  /// 8x8 mma atom consume 2xk
  /// each thread has 8xk, issue 4 mma
  for (int kk = 0; kk < 4; kk++) {
    MMA_16x8x8_F32BF16BF16F32_TN::fma(
        c_frags[0][0], c_frags[0][1], c_frags[1][0], c_frags[1][1],
        a_frags[0][kk], a_frags[1][kk], b_frags[kk], c_frags[0][0],
        c_frags[0][1], c_frags[1][0], c_frags[1][1]);
  }
  nv_bfloat162 c_frags_bf16x2[2];
  c_frags_bf16x2[0] = make_bfloat162(c_frags[0][0], c_frags[0][1]);
  c_frags_bf16x2[1] = make_bfloat162(c_frags[1][0], c_frags[1][1]);

  v_mov<uint32_t>(&out[thr_idx_mn * kCtaTileN + thr_idx_k * 2],
                  &c_frags_bf16x2[0]);
  v_mov<uint32_t>(&out[(thr_idx_mn + 8) * kCtaTileN + thr_idx_k * 2],
                  &c_frags_bf16x2[1]);
}

namespace gqmm::cxx {
void gemm_bf16_mxfp4(void *A, void *W, void *scales, void *out, int m, int n,
                     int k) {
  k_gemm_bf16_mxfp4<16, 8, 32><<<1, 32>>>(
      reinterpret_cast<nv_bfloat16 *>(A), reinterpret_cast<uint32_t *>(W),
      reinterpret_cast<uint8_t *>(scales), reinterpret_cast<nv_bfloat16 *>(out),
      m, n, k);
}
} // namespace gqmm::cxx