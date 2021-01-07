/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once
#include <ATen/ATen.h>
#include <cuda.h>

using namespace at;

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

static constexpr int32_t kWarpSize = 32;
static constexpr int32_t kMaxThreads = 1024;

struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(Half* p) {
#if CUDA_VERSION >= 9000

#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(p), "r"(__HALF2_TO_UI(a)), "r"(__HALF2_TO_UI(b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(a.x), "r"(b.x));
#endif
  }
};

template <typename T>
struct Vec4T {};

template <>
struct Vec4T<float> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const float* p) {
    acc = *((const float4*)p);
  }

  DEVICE_INLINE void store(float* p) {
    *((float4*)p) = acc;
  }

  DEVICE_INLINE void store(Half* p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }

  DEVICE_INLINE static void copy(const float* src, float* dst) {
    *((float4*)dst) = *((const float4*)src);
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(Vec4T<float> a, float b) {
    acc.x = __fmaf_rn(a.acc.x, b, acc.x);
    acc.y = __fmaf_rn(a.acc.y, b, acc.y);
    acc.z = __fmaf_rn(a.acc.z, b, acc.z);
    acc.w = __fmaf_rn(a.acc.w, b, acc.w);
  }
};

template <>
struct Vec4T<double> {
  double4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const double* p) {
    acc = *((const double4*)p);
  }

  DEVICE_INLINE void store(double* p) {
    *((double4*)p) = acc;
  }

  DEVICE_INLINE static void copy(const double* src, double* dst) {
    *((double4*)dst) = *((const double4*)src);
  }

  // this <- this + a * b
  DEVICE_INLINE void fma_(Vec4T<double> a, double b) {
    acc.x = __fma_rn(a.acc.x, b, acc.x);
    acc.y = __fma_rn(a.acc.y, b, acc.y);
    acc.z = __fma_rn(a.acc.z, b, acc.z);
    acc.w = __fma_rn(a.acc.w, b, acc.w);
  }
};

template <typename T>
DEVICE_INLINE T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val += shfl_xor(val, mask);
  }
  return val;
}

__host__ DEVICE_INLINE int32_t div_round_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}
