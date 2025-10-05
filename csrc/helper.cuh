#pragma once
#include <assert.h>
#include <cstdint>
#include <cstdio>

#if defined(__CUDA_ARCH__)
#define GQMM_INVALID_CONTROL_PATH(x)                                           \
  assert(0 && x);                                                              \
  printf(x);                                                                   \
  __brkpt()
#else
#define GQMM_INVALID_CONTROL_PATH(x)                                           \
  assert(0 && x);                                                              \
  printf(x)
#endif

struct MMA_16x8x8_F32BF16BF16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  __device__ static void fma(float &d0, float &d1, float &d2, float &d3,
                             uint32_t const &a0, uint32_t const &a1,
                             uint32_t const &b0, float const &c0,
                             float const &c1, float const &c2,
                             float const &c3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5},"
                 "{%6},"
                 "{%7,  %8,  %9,  %10};\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2),
                   "f"(c3));
#else
    GQMM_INVALID_CONTROL_PATH("Attempting to use MMA_16x8x8_F32BF16BF16F32_TN "
                              "without __CUDA_ARCH__ >= 800");
#endif
  }
};

/// @brief Vectorized move from src to dst
/// @tparam MovT The vectorized type, e.g., uint4, float4
/// @param dst Destination pointer
/// @param src Source pointer
template <typename MovT>
__device__ inline void v_mov(void *dst, const void *src) {
  *reinterpret_cast<MovT *>(dst) = *reinterpret_cast<const MovT *>(src);
}
