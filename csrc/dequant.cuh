#pragma once
#include "numeric.cuh"

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
__device__ inline nv_bfloat16 dequant_mxfp4(uint32_t q) {
  uint32_t exp_mantissa = q & 0x7;
  uint32_t sgn = q >> 3;
  uint32_t tmp = (sgn << 15) | (exp_mantissa << (7 - 1));
  auto bias = make_bfloat16_from_raw(0x7E80);
  return make_bfloat16_from_raw(tmp) * bias;
}

__device__ inline nv_bfloat16 dequant_mxfp4_with_scale(uint32_t q,
                                                       uint32_t scale) {
  auto scale_dquant = make_bfloat16_from_raw(scale << 7);
  return dequant_mxfp4(q) * scale_dquant;
}