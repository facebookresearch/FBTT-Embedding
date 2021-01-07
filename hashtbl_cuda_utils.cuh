/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <thrust/pair.h>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr uint32_t c1 = 0xcc9e2d51;
constexpr uint32_t c2 = 0x1b873593;

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

DEVICE_INLINE int64_t gpuAtomicCAS(int64_t* p, int64_t compare, int64_t val) {
  static_assert(
      sizeof(int64_t) == sizeof(unsigned long long),
      "expected int64_t to be unsigned long long");
  return static_cast<int64_t>(atomicCAS(
      reinterpret_cast<unsigned long long int*>(p),
      static_cast<unsigned long long int>(compare),
      static_cast<unsigned long long int>(val)));
}

DEVICE_INLINE int32_t gpuAtomicCAS(int32_t* p, int32_t compare, int32_t val) {
  return atomicCAS(p, compare, val);
}

__host__ DEVICE_INLINE uint32_t rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}

__host__ DEVICE_INLINE uint32_t murmor_hash_3_32(int64_t h_in, int32_t C) {
  uint32_t h = 0;
  uint32_t* ptr = reinterpret_cast<uint32_t*>(&h_in);
  uint32_t k1 = ptr[0];
  k1 *= c1;
  k1 = rotl32(k1, 15);
  k1 *= c2;
  h ^= k1;
  h = rotl32(h, 13);
  h = h * 5 + 0xe6546b64;
  uint32_t k2 = ptr[1];
  k2 *= c1;
  k2 = rotl32(k2, 15);
  k2 *= c2;
  h ^= k2;
  h = rotl32(h, 13);
  h = h * 5 + 0xe6546b64;

  h ^= 2;
  // MurmorHash3 32-bit mixing function.
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
  return ((uint64_t)h * (uint64_t)C) >> 32;
}

__host__ DEVICE_INLINE uint32_t murmor_hash_3_32(int32_t h_in, int32_t C) {
  uint32_t h = 0;
  uint32_t k = h_in;
  k *= c1;
  k = rotl32(k, 15);
  k *= c2;
  h ^= k;
  h = rotl32(h, 13);
  h = h * 5 + 0xe6546b64;

  h ^= 1;
  // MurmorHash3 32-bit mixing function.
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
  return ((uint64_t)h * (uint64_t)C) >> 32;
}

#define UNUSED_KEY -1

template <typename key_type, typename value_type, bool accumulate>
__forceinline__ __device__ int32_t hashtbl_insert(
    key_type insert_key,
    value_type insert_value,
    int32_t hashtbl_size,
    int32_t max_probes,
    key_type* __restrict__ hashtbl_keys,
    value_type* __restrict__ hashtbl_values) {
  int32_t hashtbl_idx = murmor_hash_3_32(insert_key, hashtbl_size);
  int32_t counter = 0;
  while (counter++ < max_probes) {
    key_type old_key =
        gpuAtomicCAS(&hashtbl_keys[hashtbl_idx], UNUSED_KEY, insert_key);
    if (accumulate) {
      if (UNUSED_KEY == old_key || insert_key == old_key) {
        gpuAtomicAdd(&hashtbl_values[hashtbl_idx], insert_value);
        return hashtbl_idx;
      }
    } else {
      if (UNUSED_KEY == old_key) {
        hashtbl_values[hashtbl_idx] = insert_value;
        return hashtbl_idx;
      } else if (insert_key == old_key) {
        return hashtbl_idx;
      }
    }
    // linear probe
    hashtbl_idx = (hashtbl_idx + 1) % hashtbl_size;
  }

  return -1;
}

template <typename key_type>
__forceinline__ __device__ int32_t hashtbl_find(
    key_type key,
    int32_t hashtbl_size,
    int32_t max_probes,
    const key_type* __restrict__ hashtbl_keys) {
  int32_t hashtbl_idx = murmor_hash_3_32(key, hashtbl_size);
  int32_t counter = 0;
  while (counter++ < max_probes) {
    if (key == hashtbl_keys[hashtbl_idx]) {
      return hashtbl_idx;
    } else if (UNUSED_KEY == key) {
      return -1;
    }
    // linear probe
    hashtbl_idx = (hashtbl_idx + 1) % hashtbl_size;
  }

  return -1;
}
