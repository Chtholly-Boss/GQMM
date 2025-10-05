#pragma once
#include <cstdint>
#include <cuda_bf16.h>

constexpr auto ceil_div(auto a, auto b) { return (a + b - 1) / b; }

__device__ inline nv_bfloat16 make_bfloat16_from_raw(uint16_t raw) {
  __nv_bfloat16_raw r;
  r.x = raw;
  return nv_bfloat16(r);
}