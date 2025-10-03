#include "binding.h"
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>

constexpr auto ceil_div(auto a, auto b) { return (a + b - 1) / b; }

static __device__ nv_bfloat16 make_bfloat16_from_raw(uint16_t raw) {
  __nv_bfloat16_raw r;
  r.x = raw;
  return nv_bfloat16(r);
}

/// @brief Dequantize a 4-bit MXFP number to bfloat16
/// @param q 4-bit MXFP number stored in the lower 4 bits of a uint32_t
/// @return Dequantized bfloat16 number
/// @details Equivalent to lookup table:
/// 0 -> 0.0 | 8 ->  -0.0 |
/// 1 -> 0.5 | 9 ->  -0.5 |
/// 2 -> 1.0 | 10 -> -1.0 |
/// 3 -> 1.5 | 11 -> -1.5 |
/// 4 -> 2.0 | 12 -> -2.0 |
/// 5 -> 3.0 | 13 -> -3.0 |
/// 6 -> 4.0 | 14 -> -4.0 |
/// 7 -> 6.0 | 15 -> -6.0 |
__device__ nv_bfloat16 dequant_mxfp4(uint32_t q) {
  uint32_t exp_mantissa = q & 0x7;
  uint32_t sgn = q >> 3;
  uint32_t tmp = (sgn << 15) | (exp_mantissa << (7 - 1));
  auto bias = make_bfloat16_from_raw(0x7E80);
  return make_bfloat16_from_raw(tmp) * bias;
}

__device__ nv_bfloat16 dequant_mxfp4_with_scale(uint32_t q, uint32_t scale) {
  auto scale_dquant = make_bfloat16_from_raw(scale << 7);
  return dequant_mxfp4(q) * scale_dquant;
}

static __global__ void k_dequant_mxfp4(uint32_t *blocks, uint32_t *scales,
                                       uint32_t *out, int num_blocks) {
  // each thread process a block
  auto const gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= num_blocks)
    return;

  // mxfp4 use fixed group size of 32, 1 ld.dwordx4 = 128 bits = 32 x 4-bit
  uint32_t block[4];
  auto block_ptr = reinterpret_cast<uint4 *>(blocks) + gid;
  *reinterpret_cast<uint4 *>(block) = *block_ptr;

  // 1 scale per 32 elements, 4 scales packed in a uint32_t
  auto scale_ptr = scales + gid / 4;
  uint32_t scale = (*scale_ptr >> (gid % 4 * 8)) & 0xFF;

  auto out_ptr = reinterpret_cast<nv_bfloat16 *>(out) + gid * 32;

  nv_bfloat16 v8dq[8];
  for (int i = 0; i < 4; i++) {
    uint32_t v8q = block[i];
    for (int j = 0; j < 8; j++) {
      uint32_t v4q = (v8q >> (j * 4)) & 0xF;
      v8dq[j] = dequant_mxfp4_with_scale(v4q, scale);
    }
    *reinterpret_cast<uint4 *>(out_ptr + i * 8) =
        *reinterpret_cast<uint4 *>(v8dq);
  }
}

namespace gqmm::cxx {
void dequant_mxfp4(void *blocks, void *scales, void *out, int num_blocks) {
  k_dequant_mxfp4<<<ceil_div(num_blocks, 32), 32>>>(
      reinterpret_cast<uint32_t *>(blocks),
      reinterpret_cast<uint32_t *>(scales), reinterpret_cast<uint32_t *>(out),
      num_blocks);
}
} // namespace gqmm::cxx
