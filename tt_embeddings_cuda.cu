/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <THC/THCAtomics.cuh>
#include <mutex>
#include "cub/device/device_partition.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "hashtbl_cuda_utils.cuh"
#include "tt_cuda_utils.cuh"

using namespace at;

namespace {

constexpr int32_t MAX_PROBES = 3;

enum {
  OPTIM_SGD = 0,
  OPTIM_ADAGRAD = 1,
  OPTIM_DENSE = 2,
};

} // namespace

inline void cuda_gemm_batched_fp32_fp32(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    void** a_array,
    int lda,
    void** b_array,
    int ldb,
    float* beta,
    void** c_array,
    int ldc,
    int batch_count) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());
  cublasGemmBatchedEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      a_array,
      CUDA_R_32F,
      lda,
      b_array,
      CUDA_R_32F,
      ldb,
      beta,
      c_array,
      CUDA_R_32F,
      ldc,
      batch_count,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

__global__ void init_batch_gemm_backward_2T_kernel(
    int32_t N,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const int64_t* __restrict__ L,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_output,
    int32_t* __restrict__ tt_idx,
    float** __restrict__ a0_ptr,
    float** __restrict__ b0_ptr,
    float** __restrict__ c0_ptr,
    float** __restrict__ a1_ptr,
    float** __restrict__ b1_ptr,
    float** __restrict__ c1_ptr) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto cidx = __ldg(&colidx[n]);
    auto ridx = __ldg(&rowidx[n]);
    auto tidx = __ldg(&tableidx[n]);
    int32_t tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    int32_t tt_idx_1 = cidx / L[1];
    tt_idx[0 * N + n] = tt_idx_0;
    tt_idx[1 * N + n] = tt_idx_1;
    float* d_output_ptr = (float*)&(d_output[tidx][ridx][0]);
    a0_ptr[0 * N + n] = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    b0_ptr[0 * N + n] = d_output_ptr;
    c0_ptr[0 * N + n] = (float*)&(tr_tt_cores_1[n][0]);
    a1_ptr[0 * N + n] = d_output_ptr;
    b1_ptr[0 * N + n] = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    c1_ptr[0 * N + n] = (float*)&(tr_tt_cores_0[n][0]);
  }
}

__global__ void init_batch_gemm_backward_3T_kernel(
    int32_t N,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const int64_t* __restrict__ L,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_output,
    int32_t* __restrict__ tt_idx,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ a0_ptr,
    float** __restrict__ b0_ptr,
    float** __restrict__ c0_ptr,
    float** __restrict__ a1_ptr,
    float** __restrict__ b1_ptr,
    float** __restrict__ c1_ptr) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto cidx = __ldg(&colidx[n]);
    auto ridx = __ldg(&rowidx[n]);
    auto tidx = __ldg(&tableidx[n]);
    int32_t tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    int32_t tt_idx_1 = cidx / L[1];
    cidx = cidx % L[1];
    int32_t tt_idx_2 = cidx / L[2];
    tt_idx[0 * N + n] = tt_idx_0;
    tt_idx[1 * N + n] = tt_idx_1;
    tt_idx[2 * N + n] = tt_idx_2;
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    float* d_output_ptr = (float*)&(d_output[tidx][ridx][0]);
    float* tt_cores_0_ptr = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    float* tt_cores_1_ptr = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    a_ptr[0 * N + n] = tt_cores_1_ptr;
    b_ptr[0 * N + n] = tt_cores_0_ptr;
    c_ptr[0 * N + n] = tr_0_ptr;
    a0_ptr[0 * N + n] = tt_cores_0_ptr;
    b0_ptr[0 * N + n] = tr_0_ptr;
    c0_ptr[0 * N + n] = (float*)&(tr_tt_cores_1[n][0]);
    a1_ptr[0 * N + n] = tr_0_ptr;
    b1_ptr[0 * N + n] = tt_cores_1_ptr;
    c1_ptr[0 * N + n] = (float*)&(tr_tt_cores_0[n][0]);
    a0_ptr[1 * N + n] = tr_0_ptr;
    b0_ptr[1 * N + n] = d_output_ptr;
    c0_ptr[1 * N + n] = (float*)&(tr_tt_cores_2[n][0]);
    a1_ptr[1 * N + n] = d_output_ptr;
    b1_ptr[1 * N + n] = (float*)&(tt_cores_2[tidx][tt_idx_2][0]);
    c1_ptr[1 * N + n] = tr_0_ptr;
  }
}

__global__ void init_batch_gemm_backward_4T_kernel(
    int32_t N,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const int64_t* __restrict__ L,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_3,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores_3,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_output,
    int32_t* __restrict__ tt_idx,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ a0_ptr,
    float** __restrict__ b0_ptr,
    float** __restrict__ c0_ptr,
    float** __restrict__ a1_ptr,
    float** __restrict__ b1_ptr,
    float** __restrict__ c1_ptr) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto cidx = __ldg(&colidx[n]);
    auto ridx = __ldg(&rowidx[n]);
    auto tidx = __ldg(&tableidx[n]);
    int32_t tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    int32_t tt_idx_1 = cidx / L[1];
    cidx = cidx % L[1];
    int32_t tt_idx_2 = cidx / L[2];
    cidx = cidx % L[2];
    int32_t tt_idx_3 = cidx / L[3];
    tt_idx[0 * N + n] = tt_idx_0;
    tt_idx[1 * N + n] = tt_idx_1;
    tt_idx[2 * N + n] = tt_idx_2;
    tt_idx[3 * N + n] = tt_idx_3;
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    float* tr_1_ptr = (float*)&(tr_1[n][0]);
    float* d_output_ptr = (float*)&(d_output[tidx][ridx][0]);
    float* tt_cores_0_ptr = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    float* tt_cores_1_ptr = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    float* tt_cores_2_ptr = (float*)&(tt_cores_2[tidx][tt_idx_2][0]);
    a_ptr[0 * N + n] = tt_cores_1_ptr;
    b_ptr[0 * N + n] = tt_cores_0_ptr;
    c_ptr[0 * N + n] = tr_0_ptr;
    a_ptr[1 * N + n] = tt_cores_2_ptr;
    b_ptr[1 * N + n] = tr_0_ptr;
    c_ptr[1 * N + n] = tr_1_ptr;
    a0_ptr[0 * N + n] = tt_cores_0_ptr;
    b0_ptr[0 * N + n] = tr_0_ptr;
    c0_ptr[0 * N + n] = (float*)&(tr_tt_cores_1[n][0]);
    a1_ptr[0 * N + n] = b0_ptr[0 * N + n];
    b1_ptr[0 * N + n] = tt_cores_1_ptr;
    c1_ptr[0 * N + n] = (float*)&(tr_tt_cores_0[n][0]);
    a0_ptr[1 * N + n] = tr_0_ptr;
    b0_ptr[1 * N + n] = tr_1_ptr;
    c0_ptr[1 * N + n] = (float*)&(tr_tt_cores_2[n][0]);
    a1_ptr[1 * N + n] = b0_ptr[1 * N + n];
    b1_ptr[1 * N + n] = tt_cores_2_ptr;
    c1_ptr[1 * N + n] = tr_0_ptr;
    a0_ptr[2 * N + n] = tr_1_ptr;
    b0_ptr[2 * N + n] = d_output_ptr;
    c0_ptr[2 * N + n] = (float*)&(tr_tt_cores_3[n][0]);
    a1_ptr[2 * N + n] = d_output_ptr;
    b1_ptr[2 * N + n] = (float*)&(tt_cores_3[tidx][tt_idx[3 * N + n]][0]);
    c1_ptr[2 * N + n] = tr_1_ptr;
  }
}

void init_batch_gemm_backward_cuda(
    int32_t T,
    int32_t N,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const int64_t* __restrict__ L,
    const std::vector<Tensor>& tt_cores,
    const std::vector<Tensor>& tr_tt_cores,
    const std::vector<Tensor>& tr,
    Tensor d_output,
    int32_t* __restrict__ tt_idx,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr,
    float** __restrict__ a0_ptr,
    float** __restrict__ b0_ptr,
    float** __restrict__ c0_ptr,
    float** __restrict__ a1_ptr,
    float** __restrict__ b1_ptr,
    float** __restrict__ c1_ptr) {
  int32_t threads = (N > 256 ? 256 : 32);
  int32_t num_blocks = (N + threads - 1) / threads;
  if (T == 2) {
    init_batch_gemm_backward_2T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        colidx,
        rowidx,
        tableidx,
        L,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        d_output.packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_idx,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (T == 3) {
    init_batch_gemm_backward_3T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        colidx,
        rowidx,
        tableidx,
        L,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        d_output.packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_idx,
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (T == 4) {
    init_batch_gemm_backward_4T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        colidx,
        rowidx,
        tableidx,
        L,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[3].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr_tt_cores[3].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        d_output.packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_idx,
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

__global__ void update_d_tt_cores_kernel(
    int32_t N,
    int32_t D,
    const int32_t* __restrict__ tt_idx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_tt_cores,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_tt_cores) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n < N) {
    auto idx = __ldg(&tt_idx[n]);
    auto tidx = __ldg(&tableidx[n]);
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      atomicAdd(&(d_tt_cores[tidx][idx][d]), tr_tt_cores[n][d]);
    }
  }
}

__global__ void update_tt_cores_sgd_kernel(
    int32_t B,
    int32_t D,
    int32_t num_tables,
    float learning_rate,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_tt_cores,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  for (int32_t i = 0; i < num_tables; i++) {
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      tt_cores[i][b][d] -= learning_rate * d_tt_cores[i][b][d];
    }
  }
}

__global__ void update_tt_cores_adagrad_kernel(
    int32_t B,
    int32_t D,
    int32_t num_tables,
    float learning_rate,
    float eps,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> d_tt_cores,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> optimizer_state,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b >= B) {
    return;
  }
  for (int32_t i = 0; i < num_tables; i++) {
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      optimizer_state[i][b][d] += d_tt_cores[i][b][d] * d_tt_cores[i][b][d];
      tt_cores[i][b][d] -= learning_rate * d_tt_cores[i][b][d] /
          (sqrt(optimizer_state[i][b][d]) + eps);
    }
  }
}

std::vector<Tensor> tt_embeddings_backward_cuda(
    int32_t optim,
    int32_t batch_count,
    int32_t D,
    float learning_rate,
    float eps,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    Tensor d_output,
    c10::optional<std::vector<Tensor>> optimizer_state,
    std::vector<Tensor>& tt_cores) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(d_output.get_device());
  int32_t T = tt_p_shapes.size();
  int32_t num_tables = tt_cores[0].size(0);

  std::vector<Tensor> d_tt_cores;
  std::vector<Tensor> tr_tt_cores;
  for (int32_t t = 0; t < T; ++t) {
    d_tt_cores.push_back(at::zeros_like(tt_cores[t]));
    tr_tt_cores.push_back(
        at::empty({batch_count, tt_cores[t].size(2)}, tt_cores[t].options()));
  }
  if (nnz == 0) {
    return d_tt_cores;
  }

  // batch gemm parameters
  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0];
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_;
    k[t] = tt_ranks[t + 1];
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2];
    m_ = m_ * tt_q_shapes[t + 1];
  }
  // allocate the immediate buffers
  std::vector<Tensor> tr;

  int64_t tr_size = tt_q_shapes[0] * tt_ranks[1];
  for (int32_t t = 0; t < T - 2; ++t) {
    tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
    tr.push_back(at::empty({batch_count, tr_size}, tt_cores[0].options()));
  }

  auto tt_idx =
      at::empty({T * batch_count}, tt_cores[0].options().dtype(at::kInt));
  auto a_ptr_tensor = at::empty(
      {(T - 2) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty(
      {(T - 2) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty(
      {(T - 2) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
  auto a0_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b0_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c0_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a0_ptr = (float**)a0_ptr_tensor.data_ptr<int64_t>();
  float** b0_ptr = (float**)b0_ptr_tensor.data_ptr<int64_t>();
  float** c0_ptr = (float**)c0_ptr_tensor.data_ptr<int64_t>();
  auto a1_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b1_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c1_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a1_ptr = (float**)a1_ptr_tensor.data_ptr<int64_t>();
  float** b1_ptr = (float**)b1_ptr_tensor.data_ptr<int64_t>();
  float** c1_ptr = (float**)c1_ptr_tensor.data_ptr<int64_t>();
  for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count) {
    int32_t end_idx =
        start_idx + batch_count < nnz ? start_idx + batch_count : nnz;
    int32_t N = end_idx - start_idx;
    init_batch_gemm_backward_cuda(
        T,
        N,
        &(colidx.data_ptr<int64_t>()[start_idx]),
        &(rowidx.data_ptr<int64_t>()[start_idx]),
        &(tableidx.data_ptr<int64_t>()[start_idx]),
        L.data_ptr<int64_t>(),
        tt_cores,
        tr_tt_cores,
        tr,
        d_output,
        tt_idx.data_ptr<int32_t>(),
        a_ptr,
        b_ptr,
        c_ptr,
        a0_ptr,
        b0_ptr,
        c0_ptr,
        a1_ptr,
        b1_ptr,
        c1_ptr);
    // recompute forward
    for (int32_t t = 0; t < T - 2; ++t) {
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n[t],
          m[t],
          k[t],
          &alpha,
          (void**)&(a_ptr[t * N]),
          n[t],
          (void**)&(b_ptr[t * N]),
          k[t],
          &beta,
          (void**)&(c_ptr[t * N]),
          n[t],
          N);
    } // for (int32_t t = 0; t < T - 2; ++t)
    // backward propagation
    for (int32_t t = T - 2; t >= 0; --t) {
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          n[t],
          k[t],
          m[t],
          &alpha,
          (void**)&(b0_ptr[t * N]),
          n[t],
          (void**)&(a0_ptr[t * N]),
          k[t],
          &beta,
          (void**)&(c0_ptr[t * N]),
          n[t],
          N);
      int32_t D_0 = tt_cores[t + 1].size(2);
      int32_t tx_0 = std::min(1024, D_0);
      int32_t ty_0 = 1024 / tx_0;
      update_d_tt_cores_kernel<<<
          div_round_up(N, ty_0),
          dim3(tx_0, ty_0),
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          N,
          D_0,
          &(tt_idx.data_ptr<int32_t>()[(t + 1) * N]),
          &(tableidx.data_ptr<int64_t>()[start_idx]),
          tr_tt_cores[t + 1].packed_accessor32<float, 2, RestrictPtrTraits>(),
          d_tt_cores[t + 1].packed_accessor32<float, 3, RestrictPtrTraits>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          k[t],
          m[t],
          n[t],
          &alpha,
          (void**)&(b1_ptr[t * N]),
          n[t],
          (void**)&(a1_ptr[t * N]),
          n[t],
          &beta,
          (void**)&(c1_ptr[t * N]),
          k[t],
          N);
      if (t == 0) {
        int32_t D_1 = tt_cores[0].size(2);
        int32_t tx_1 = std::min(1024, D_1);
        int32_t ty_1 = 1024 / tx_1;
        update_d_tt_cores_kernel<<<
            div_round_up(N, ty_1),
            dim3(tx_1, ty_1),
            0,
            c10::cuda::getCurrentCUDAStream()>>>(
            N,
            D_1,
            &(tt_idx.data_ptr<int32_t>()[t * N]),
            &(tableidx.data_ptr<int64_t>()[start_idx]),
            tr_tt_cores[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
            d_tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } // for (int32_t t = T - 2; t >=0 ; --t)
  } // for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count)
  if (optim == OPTIM_ADAGRAD) {
    for (int32_t t = 0; t < T; ++t) {
      int32_t y_size = tt_cores[t].size(1);
      int32_t x_size = tt_cores[t].size(2);
      int32_t tx = std::min(1024, y_size);
      int32_t ty = 1024 / tx;
      update_tt_cores_adagrad_kernel<<<
          div_round_up(x_size, ty),
          dim3(tx, ty),
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          y_size,
          x_size,
          num_tables,
          learning_rate,
          eps,
          d_tt_cores[t].packed_accessor32<float, 3, RestrictPtrTraits>(),
          (*optimizer_state)[t]
              .packed_accessor32<float, 3, RestrictPtrTraits>(),
          tt_cores[t].packed_accessor32<float, 3, RestrictPtrTraits>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  } else if (optim == OPTIM_SGD) {
    for (int32_t t = 0; t < T; ++t) {
      int32_t y_size = tt_cores[t].size(1);
      int32_t x_size = tt_cores[t].size(2);
      int32_t tx = std::min(1024, y_size);
      int32_t ty = 1024 / tx;
      update_tt_cores_sgd_kernel<<<
          div_round_up(x_size, ty),
          dim3(tx, ty),
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          y_size,
          x_size,
          num_tables,
          learning_rate,
          d_tt_cores[t].packed_accessor32<float, 3, RestrictPtrTraits>(),
          tt_cores[t].packed_accessor32<float, 3, RestrictPtrTraits>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  return d_tt_cores;
}

std::vector<Tensor> tt_embeddings_backward_dense_cuda(
    int32_t batch_count,
    int32_t D,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores) {
  return tt_embeddings_backward_cuda(
      OPTIM_DENSE,
      batch_count,
      D,
      0.0,
      0.0,
      tt_p_shapes,
      tt_q_shapes,
      tt_ranks,
      L,
      nnz,
      colidx,
      rowidx,
      tableidx,
      d_output,
      c10::nullopt,
      tt_cores);
}

void tt_embeddings_backward_sgd_cuda(
    int32_t batch_count,
    int32_t D,
    float learning_rate,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores) {
  tt_embeddings_backward_cuda(
      OPTIM_SGD,
      batch_count,
      D,
      learning_rate,
      0.0,
      tt_p_shapes,
      tt_q_shapes,
      tt_ranks,
      L,
      nnz,
      colidx,
      rowidx,
      tableidx,
      d_output,
      c10::nullopt,
      tt_cores);
}

void tt_embeddings_backward_adagrad_cuda(
    int32_t batch_count,
    int32_t D,
    float learning_rate,
    float eps,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& optimizer_state,
    std::vector<Tensor>& tt_cores) {
  tt_embeddings_backward_cuda(
      OPTIM_ADAGRAD,
      batch_count,
      D,
      learning_rate,
      eps,
      tt_p_shapes,
      tt_q_shapes,
      tt_ranks,
      L,
      nnz,
      colidx,
      rowidx,
      tableidx,
      d_output,
      optimizer_state,
      tt_cores);
}

__global__ void init_batch_gemm_forward_2T_kernel(
    int N,
    const int64_t* __restrict__ L,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto tidx = __ldg(&tableidx[n]);
    auto cidx = __ldg(&colidx[n]);
    auto tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    auto tt_idx_1 = cidx / L[1];
    a_ptr[0 * N + n] = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    b_ptr[0 * N + n] = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    c_ptr[0 * N + n] = (float*)&(tr_0[n][0]);
  }
}

__global__ void init_batch_gemm_forward_3T_kernel(
    int N,
    const int64_t* __restrict__ L,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_1,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto tidx = __ldg(&tableidx[n]);
    auto cidx = __ldg(&colidx[n]);
    auto tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    auto tt_idx_1 = cidx / L[1];
    cidx = cidx % L[1];
    auto tt_idx_2 = cidx / L[2];
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    a_ptr[0 * N + n] = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    b_ptr[0 * N + n] = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    c_ptr[0 * N + n] = tr_0_ptr;
    a_ptr[1 * N + n] = (float*)&(tt_cores_2[tidx][tt_idx_2][0]);
    b_ptr[1 * N + n] = tr_0_ptr;
    c_ptr[1 * N + n] = (float*)&(tr_1[n][0]);
  }
}

__global__ void init_batch_gemm_forward_4T_kernel(
    int N,
    const int64_t* __restrict__ L,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ tableidx,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_0,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_1,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_2,
    PackedTensorAccessor32<float, 3, RestrictPtrTraits> tt_cores_3,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_0,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_1,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> tr_2,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    auto tidx = __ldg(&tableidx[n]);
    auto cidx = __ldg(&colidx[n]);
    auto tt_idx_0 = cidx / L[0];
    cidx = cidx % L[0];
    auto tt_idx_1 = cidx / L[1];
    cidx = cidx % L[1];
    auto tt_idx_2 = cidx / L[2];
    cidx = cidx % L[2];
    auto tt_idx_3 = cidx / L[3];
    float* tr_0_ptr = (float*)&(tr_0[n][0]);
    float* tr_1_ptr = (float*)&(tr_1[n][0]);
    a_ptr[0 * N + n] = (float*)&(tt_cores_1[tidx][tt_idx_1][0]);
    b_ptr[0 * N + n] = (float*)&(tt_cores_0[tidx][tt_idx_0][0]);
    c_ptr[0 * N + n] = tr_0_ptr;
    a_ptr[1 * N + n] = (float*)&(tt_cores_2[tidx][tt_idx_2][0]);
    b_ptr[1 * N + n] = tr_0_ptr;
    c_ptr[1 * N + n] = tr_1_ptr;
    a_ptr[2 * N + n] = (float*)&(tt_cores_3[tidx][tt_idx_3][0]);
    b_ptr[2 * N + n] = tr_1_ptr;
    c_ptr[2 * N + n] = (float*)&(tr_2[n][0]);
  }
}

void init_batch_gemm_forward_cuda(
    int32_t T,
    int32_t N,
    const int64_t* __restrict__ L,
    const int64_t* __restrict__ colidx,
    const int64_t* __restrict__ tableidx,
    const std::vector<Tensor>& tt_cores,
    const std::vector<Tensor>& tr,
    float** __restrict__ a_ptr,
    float** __restrict__ b_ptr,
    float** __restrict__ c_ptr) {
  int32_t threads = (N > 256 ? 256 : 32);
  int32_t num_blocks = (N + threads - 1) / threads;
  if (T == 2) {
    init_batch_gemm_forward_2T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        L,
        colidx,
        tableidx,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        a_ptr,
        b_ptr,
        c_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (T == 3) {
    init_batch_gemm_forward_3T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        L,
        colidx,
        tableidx,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        a_ptr,
        b_ptr,
        c_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (T == 4) {
    init_batch_gemm_forward_4T_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        L,
        colidx,
        tableidx,
        tt_cores[0].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[1].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[2].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tt_cores[3].packed_accessor32<float, 3, RestrictPtrTraits>(),
        tr[0].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[1].packed_accessor32<float, 2, RestrictPtrTraits>(),
        tr[2].packed_accessor32<float, 2, RestrictPtrTraits>(),
        a_ptr,
        b_ptr,
        c_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

__global__ void reduce_output_kernel(
    int32_t N,
    int32_t B,
    int32_t D,
    const int64_t* __restrict__ rowidx,
    const int64_t* __restrict__ tableidx,
    const float* __restrict__ tr_last,
    float* __restrict__ output) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= N) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id] ||
       tableidx[indice_id - 1] != tableidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t row_index = rowidx[indice_id];
  int64_t table_index = tableidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < N && rowidx[indice_id + SL] == row_index &&
         tableidx[indice_id + SL] == table_index) {
    SL += 1;
  }
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<float> sum(&output[table_index * B * D + row_index * D + d * 4]);
    for (int32_t sl = 0; sl < SL; ++sl) {
      Vec4T<float> tr(&tr_last[(indice_id + sl) * D + d * 4]);
      sum.acc.x += tr.acc.x;
      sum.acc.y += tr.acc.y;
      sum.acc.z += tr.acc.z;
      sum.acc.w += tr.acc.w;
    }
    sum.store(&output[table_index * B * D + row_index * D + d * 4]);
  }
}

Tensor tt_embeddings_forward_cuda(
    int32_t batch_count,
    int32_t num_tables,
    int32_t B,
    int32_t D,
    const std::vector<int>& tt_p_shapes,
    const std::vector<int>& tt_q_shapes,
    const std::vector<int>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor colidx,
    Tensor rowidx,
    Tensor tableidx,
    const std::vector<Tensor>& tt_cores) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(rowidx.get_device());
  int32_t T = tt_p_shapes.size();
  auto output =
      at::zeros({num_tables, B, D}, tt_cores[0].options().dtype(at::kFloat));
  if (nnz == 0) {
    return output;
  }

  TORCH_CHECK(batch_count > 0);
  TORCH_CHECK(D > 0);
  TORCH_CHECK(D % 4 == 0);
  TORCH_CHECK(T > 0);

  // batch gemm parameters
  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0];
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_;
    k[t] = tt_ranks[t + 1];
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2];
    m_ = m_ * tt_q_shapes[t + 1];
  }

  // allocate the immediate buffers
  std::vector<Tensor> tr;
  int32_t tr_size = tt_q_shapes[0] * tt_ranks[1];
  for (int32_t t = 0; t < T - 1; ++t) {
    tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
    tr.push_back(at::empty(
        {batch_count, tr_size}, tt_cores[0].options().dtype(at::kFloat)));
  }
  auto a_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();
  for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count) {
    int32_t end_idx =
        start_idx + batch_count < nnz ? start_idx + batch_count : nnz;
    int32_t N = end_idx - start_idx;
    init_batch_gemm_forward_cuda(
        T,
        N,
        L.data_ptr<int64_t>(),
        &(colidx.data_ptr<int64_t>()[start_idx]),
        &(tableidx.data_ptr<int64_t>()[start_idx]),
        tt_cores,
        tr,
        a_ptr,
        b_ptr,
        c_ptr);
    // batched GEMM
    for (int32_t t = 0; t < T - 1; ++t) {
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n[t],
          m[t],
          k[t],
          &alpha,
          (void**)&(a_ptr[t * N]),
          n[t],
          (void**)&(b_ptr[t * N]),
          k[t],
          &beta,
          (void**)&(c_ptr[t * N]),
          n[t],
          N);
    }
    int32_t tx = kWarpSize;
    int32_t ty = 1024 / tx;
    dim3 threads(tx, ty);
    int32_t num_blocks = (N + ty - 1) / ty;
    reduce_output_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        B,
        D,
        &(rowidx.data_ptr<int64_t>()[start_idx]),
        &(tableidx.data_ptr<int64_t>()[start_idx]),
        tr[T - 2].data_ptr<float>(),
        output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } // for (int start_idx = 0; start_idx < nnz; start_idx += batch_count)

  return output;
}

__global__ void update_cache_state_kernel(
    int N,
    const int64_t* __restrict__ colidx,
    int32_t hashtbl_size,
    int64_t* __restrict__ hashtbl,
    int64_t* __restrict__ cache_freq) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int64_t cidx = __ldg(&colidx[n]);
    hashtbl_insert<int64_t, int64_t, true>(
        cidx, 1, hashtbl_size, MAX_PROBES, hashtbl, cache_freq);
  }
}

void update_cache_state_cuda(Tensor colidx, Tensor hashtbl, Tensor cache_freq) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(colidx.get_device());
  int32_t nnz = colidx.numel();
  if (nnz == 0) {
    return;
  }

  TORCH_CHECK(hashtbl.numel() > 0);
  TORCH_CHECK(hashtbl.numel() == cache_freq.numel());
  int32_t threads = (nnz > 256 ? 256 : 32);
  int32_t num_blocks = (nnz + threads - 1) / threads;
  update_cache_state_kernel<<<
      num_blocks,
      threads,
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      nnz,
      colidx.data_ptr<int64_t>(),
      hashtbl.numel(),
      hashtbl.data_ptr<int64_t>(),
      cache_freq.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void mark_popular_colidx_kernel(
    int32_t hashtbl_size,
    int32_t cache_size,
    int64_t* __restrict__ cache_freq_sorted_hashtbl,
    int64_t* __restrict__ hashtbl,
    int64_t* __restrict__ cache_freq,
    int32_t* __restrict__ cache_state) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= hashtbl_size) {
    return;
  }
  if (cache_freq_sorted_hashtbl[n] != -1) {
    int32_t hashtbl_idx = hashtbl_find(
        cache_freq_sorted_hashtbl[n], hashtbl_size, MAX_PROBES, hashtbl);
    if (n < cache_size) {
      cache_state[hashtbl_idx] = n;
    } else {
      hashtbl[hashtbl_idx] = -1;
      cache_freq[hashtbl_idx] = 0;
    }
  } else if (n < cache_size) {
    // a hack to use batch gemm
    cache_freq_sorted_hashtbl[n] = 0;
  }
}

__global__ void copy_output_kernel(
    int32_t N,
    int32_t D,
    int32_t start_idx,
    const float* __restrict__ tr_last,
    float* __restrict__ output) {
  int32_t n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n < N) {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<float> tr(&tr_last[n * D + d * 4]);
      tr.store(&output[(start_idx + n) * D + d * 4]);
    }
  }
}

void prefetch_cached_weights_cuda(
    int32_t batch_count,
    const std::vector<int>& tt_p_shapes,
    const std::vector<int>& tt_q_shapes,
    const std::vector<int>& tt_ranks,
    const std::vector<Tensor>& tt_cores,
    Tensor L,
    Tensor cache_freq_sorted_hashtbl,
    Tensor cache_weight) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cache_weight.get_device());
  int32_t nnz = cache_weight.size(0);
  if (nnz == 0) {
    return;
  }
  int32_t T = tt_p_shapes.size();
  int32_t D = cache_weight.size(1);
  TORCH_CHECK(batch_count > 0);
  TORCH_CHECK(D > 0);
  TORCH_CHECK(D % 4 == 0);
  TORCH_CHECK(T > 0);
  // batch gemm parameters
  std::vector<int32_t> m(T - 1);
  std::vector<int32_t> n(T - 1);
  std::vector<int32_t> k(T - 1);
  float alpha = 1.0;
  float beta = 0.0;
  int32_t m_ = tt_q_shapes[0];
  for (int32_t t = 0; t < T - 1; ++t) {
    m[t] = m_;
    k[t] = tt_ranks[t + 1];
    n[t] = tt_q_shapes[t + 1] * tt_ranks[t + 2];
    m_ = m_ * tt_q_shapes[t + 1];
  }
  // allocate the immediate buffers
  std::vector<Tensor> tr;
  int32_t tr_size = tt_q_shapes[0] * tt_ranks[1];
  for (int32_t t = 0; t < T - 1; ++t) {
    tr_size = tr_size * tt_q_shapes[t + 1] * tt_ranks[t + 2] / tt_ranks[t + 1];
    tr.push_back(at::empty(
        {batch_count, tr_size}, tt_cores[0].options().dtype(at::kFloat)));
  }
  auto a_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto b_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  auto c_ptr_tensor = at::empty(
      {(T - 1) * batch_count}, tt_cores[0].options().dtype(at::kLong));
  float** a_ptr = (float**)a_ptr_tensor.data_ptr<int64_t>();
  float** b_ptr = (float**)b_ptr_tensor.data_ptr<int64_t>();
  float** c_ptr = (float**)c_ptr_tensor.data_ptr<int64_t>();

  Tensor tableidx = zeros_like(cache_freq_sorted_hashtbl);

  for (int32_t start_idx = 0; start_idx < nnz; start_idx += batch_count) {
    int32_t end_idx =
        start_idx + batch_count < nnz ? start_idx + batch_count : nnz;
    int32_t N = end_idx - start_idx;
    init_batch_gemm_forward_cuda(
        T,
        N,
        L.data_ptr<int64_t>(),
        &(cache_freq_sorted_hashtbl.data_ptr<int64_t>()[start_idx]),
        &(tableidx.data_ptr<int64_t>()[start_idx]),
        tt_cores,
        tr,
        a_ptr,
        b_ptr,
        c_ptr);
    // batched GEMM
    for (int32_t t = 0; t < T - 1; ++t) {
      cuda_gemm_batched_fp32_fp32(
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n[t],
          m[t],
          k[t],
          &alpha,
          (void**)&(a_ptr[t * N]),
          n[t],
          (void**)&(b_ptr[t * N]),
          k[t],
          &beta,
          (void**)&(c_ptr[t * N]),
          n[t],
          N);
    }
    int32_t tx = std::min(1024, D / 4);
    int32_t ty = 1024 / tx;
    dim3 threads(tx, ty);
    int32_t num_blocks = (N + ty - 1) / ty;
    copy_output_kernel<<<
        num_blocks,
        threads,
        0,
        c10::cuda::getCurrentCUDAStream()>>>(
        N,
        D,
        start_idx,
        tr[T - 2].data_ptr<float>(),
        cache_weight.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } // for (int start_idx = 0; start_idx < nnz; start_idx += batch_count)
}

void cache_populate_cuda(
    int64_t num_embeddings,
    const std::vector<int>& tt_p_shapes,
    const std::vector<int>& tt_q_shapes,
    const std::vector<int>& tt_ranks,
    const std::vector<Tensor>& tt_cores,
    Tensor L,
    Tensor hashtbl,
    Tensor cache_freq,
    Tensor cache_state,
    Tensor cache_weight) {
  TORCH_CHECK(hashtbl.numel() > 0);
  TORCH_CHECK(hashtbl.numel() == cache_freq.numel());
  TORCH_CHECK(cache_freq.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(hashtbl.numel() >= cache_weight.size(0));

  auto cache_freq_sorted_hashtbl = empty_like(hashtbl);
  // Sort hash_table by cache_freq
  {
    auto sorted_cache_freq = empty_like(cache_freq);
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        temp_storage_bytes,
        cache_freq.data_ptr<int64_t>(),
        sorted_cache_freq.data_ptr<int64_t>(),
        hashtbl.data_ptr<int64_t>(),
        cache_freq_sorted_hashtbl.data_ptr<int64_t>(),
        cache_freq.numel(),
        0,
        sizeof(int64_t) * 8,
        at::cuda::getCurrentCUDAStream(),
        false));
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        hashtbl.options().dtype(kByte));
    AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        cache_freq.data_ptr<int64_t>(),
        sorted_cache_freq.data_ptr<int64_t>(),
        hashtbl.data_ptr<int64_t>(),
        cache_freq_sorted_hashtbl.data_ptr<int64_t>(),
        cache_freq.numel(),
        0,
        sizeof(int64_t) * 8,
        at::cuda::getCurrentCUDAStream(),
        false));
  }

  // Mark popular colidx
  int32_t hashtbl_size = hashtbl.numel();
  int32_t threads = 256;
  int32_t num_blocks = (hashtbl_size + threads - 1) / threads;
  mark_popular_colidx_kernel<<<
      num_blocks,
      threads,
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      hashtbl_size,
      cache_weight.size(0),
      cache_freq_sorted_hashtbl.data_ptr<int64_t>(),
      hashtbl.data_ptr<int64_t>(),
      cache_freq.data_ptr<int64_t>(),
      cache_state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int32_t batch_count = 200;
  prefetch_cached_weights_cuda(
      batch_count,
      tt_p_shapes,
      tt_q_shapes,
      tt_ranks,
      tt_cores,
      L,
      cache_freq_sorted_hashtbl,
      cache_weight);
}

__global__ void compute_rowidx_kernel(
    int32_t B,
    int32_t num_tables,
    const int64_t* __restrict__ offsets,
    int64_t* __restrict__ rowidx,
    int64_t* __restrict__ tableidx) {
  int32_t b = blockIdx.x * blockDim.y + threadIdx.y;
  if (b < B * num_tables) {
    int64_t colidx_start = offsets[b];
    int64_t colidx_end = offsets[b + 1];
    int32_t L = colidx_end - colidx_start;
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
      rowidx[l + colidx_start] = b % B;
      tableidx[l + colidx_start] = b / B;
    }
  }
}

__global__ void cache_lookup_kernel(
    int32_t N,
    const int64_t* __restrict__ colidx,
    int32_t hashtbl_size,
    const int64_t* __restrict__ hashtbl,
    const int32_t* __restrict__ cache_state,
    bool* __restrict__ is_tt,
    int32_t* __restrict__ cache_location) {
  int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int32_t hashtbl_idx =
        hashtbl_find(colidx[n], hashtbl_size, MAX_PROBES, hashtbl);
    if (hashtbl_idx != -1 && cache_state[hashtbl_idx] != -1) {
      is_tt[n] = false;
      cache_location[n] = cache_state[hashtbl_idx];
    } else {
      is_tt[n] = true;
    }
  }
}

std::tuple<Tensor, Tensor, Tensor, int32_t, c10::optional<Tensor>>
preprocess_indices_sync_cuda(
    Tensor colidx,
    Tensor offsets,
    int32_t num_tables,
    bool warmup,
    Tensor hashtbl,
    Tensor cache_state) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(colidx.get_device());
  auto rowidx = empty_like(colidx);
  auto tableidx = empty_like(colidx);
  if (rowidx.numel() == 0) {
    return {colidx, rowidx, tableidx, rowidx.numel(), c10::nullopt};
  }

  int32_t B = (offsets.numel() - 1) / num_tables;
  int32_t N = colidx.numel();
  int32_t num_rows = offsets.numel() - 1;

  int32_t tx = 8;
  int32_t ty = 32;
  compute_rowidx_kernel<<<
      div_round_up(num_rows, ty),
      dim3(tx, ty),
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      B,
      num_tables,
      offsets.data_ptr<int64_t>(),
      rowidx.data_ptr<int64_t>(),
      tableidx.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (warmup || num_tables != 1) {
    // if in warmup phase or num_tables != 1, we do not lookup cache
    return {colidx, rowidx, tableidx, rowidx.numel(), c10::nullopt};
  } else {
    auto partitioned_colidx = empty_like(colidx);
    auto partitioned_rowidx = empty_like(rowidx);
    auto num_tt_indices = zeros({1}, rowidx.options().dtype(kInt));
    auto cache_locations = empty_like(rowidx, rowidx.options().dtype(kInt));
    auto partitioned_cache_locations =
        empty_like(rowidx, rowidx.options().dtype(kInt));
    {
      auto is_tt = empty_like(rowidx, rowidx.options().dtype(kBool));
      int32_t threads = 256;
      int32_t num_blocks = div_round_up(N, threads);
      cache_lookup_kernel<<<
          num_blocks,
          threads,
          0,
          c10::cuda::getCurrentCUDAStream()>>>(
          N,
          colidx.data_ptr<int64_t>(),
          hashtbl.numel(),
          hashtbl.data_ptr<int64_t>(),
          cache_state.data_ptr<int32_t>(),
          is_tt.data_ptr<bool>(),
          cache_locations.data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      size_t temp_storage_bytes = 0;
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          nullptr,
          temp_storage_bytes,
          rowidx.data_ptr<int64_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_rowidx.data_ptr<int64_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          rowidx.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
      auto temp_storage = at::empty(
          {static_cast<int64_t>(temp_storage_bytes)},
          hashtbl.options().dtype(kByte));
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          rowidx.data_ptr<int64_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_rowidx.data_ptr<int64_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          rowidx.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          colidx.data_ptr<int64_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_colidx.data_ptr<int64_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          colidx.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
      AT_CUDA_CHECK(cub::DevicePartition::Flagged(
          temp_storage.data_ptr(),
          temp_storage_bytes,
          cache_locations.data_ptr<int32_t>(),
          is_tt.data_ptr<bool>(),
          partitioned_cache_locations.data_ptr<int32_t>(),
          num_tt_indices.data_ptr<int32_t>(),
          cache_locations.numel(),
          at::cuda::getCurrentCUDAStream(),
          false));
    }
    int32_t N_tt_indices;
    cudaMemcpyAsync(
        &N_tt_indices,
        num_tt_indices.data_ptr<int32_t>(),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    return {
        partitioned_colidx,
        partitioned_rowidx,
        tableidx,
        N_tt_indices,
        partitioned_cache_locations};
  }
}

__global__ void cache_forward_kernel(
    int32_t nnz,
    int32_t D,
    const int64_t* __restrict__ rowidx,
    const int32_t* __restrict__ cache_locations,
    const float* __restrict__ cache_weight,
    float* __restrict__ output) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= nnz) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t row_index = rowidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < nnz && rowidx[indice_id + SL] == row_index) {
    SL += 1;
  }

  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<float> sum(&output[row_index * D + d * 4]);
    for (int32_t sl = 0; sl < SL; ++sl) {
      int32_t idx = __ldg(&cache_locations[indice_id + sl]);
      Vec4T<float> weight(&cache_weight[idx * D + d * 4]);
      sum.acc.x += weight.acc.x;
      sum.acc.y += weight.acc.y;
      sum.acc.z += weight.acc.z;
      sum.acc.w += weight.acc.w;
    }
    sum.store(&output[row_index * D + d * 4]);
  }
}

void cache_forward_cuda(
    int32_t B,
    int32_t nnz,
    Tensor cache_locations,
    Tensor rowidx,
    Tensor cache_weight,
    Tensor output) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(rowidx.get_device());
  TORCH_CHECK(B > 0);
  int32_t D = cache_weight.size(1);
  TORCH_CHECK(D > 0);
  TORCH_CHECK(D % 4 == 0);
  if (nnz == 0) {
    return;
  }

  int32_t tx = kWarpSize;
  int32_t ty = 1024 / tx;
  dim3 threads(tx, ty);
  int32_t num_blocks = (nnz + ty - 1) / ty;
  cache_forward_kernel<<<
      num_blocks,
      threads,
      0,
      c10::cuda::getCurrentCUDAStream()>>>(
      nnz,
      D,
      rowidx.data_ptr<int64_t>(),
      cache_locations.data_ptr<int32_t>(),
      cache_weight.data_ptr<float>(),
      output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void cache_backward_sgd_kernel(
    int32_t nnz,
    int32_t D,
    const float* __restrict__ grad_output,
    const int32_t* __restrict__ cache_locations,
    const int64_t* __restrict__ rowidx,
    float learning_rate,
    float* __restrict__ cache_weight) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= nnz) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t row_index = rowidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < nnz && rowidx[indice_id + SL] == row_index) {
    SL += 1;
  }
  for (int32_t sl = 0; sl < SL; ++sl) {
    int32_t idx = __ldg(&cache_locations[indice_id + sl]);
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<acc_type<float, true>> grad_out_vec(
          &grad_output[row_index * D + d * 4]);
      gpuAtomicAdd(
          &cache_weight[idx * D + d * 4 + 0],
          -grad_out_vec.acc.x * learning_rate);
      gpuAtomicAdd(
          &cache_weight[idx * D + d * 4 + 1],
          -grad_out_vec.acc.y * learning_rate);
      gpuAtomicAdd(
          &cache_weight[idx * D + d * 4 + 2],
          -grad_out_vec.acc.z * learning_rate);
      gpuAtomicAdd(
          &cache_weight[idx * D + d * 4 + 3],
          -grad_out_vec.acc.w * learning_rate);
    }
  }
}

void cache_backward_sgd_cuda(
    int32_t nnz,
    Tensor grad_output,
    Tensor cache_locations,
    Tensor rowidx,
    float learning_rate,
    Tensor cache_weight) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cache_weight.get_device());
  if (nnz == 0) {
    return;
  }

  const auto D = cache_weight.size(1);
  TORCH_CHECK(D > 0);
  TORCH_CHECK(D % 4 == 0);
  int32_t tx = kWarpSize;
  int32_t ty = 1024 / tx;
  dim3 threads(tx, ty);
  int32_t num_blocks = div_round_up(nnz, ty);
  cache_backward_sgd_kernel<<<
      num_blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      nnz,
      D,
      grad_output.data_ptr<float>(),
      cache_locations.data_ptr<int32_t>(),
      rowidx.data_ptr<int64_t>(),
      learning_rate,
      cache_weight.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return;
}

__global__ void cache_backward_dense_kernel(
    int32_t nnz,
    int32_t D,
    const float* __restrict__ grad_output,
    const int32_t* __restrict__ cache_locations,
    const int64_t* __restrict__ rowidx,
    float* __restrict__ grad_cache_weight) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= nnz) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t row_index = rowidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < nnz && rowidx[indice_id + SL] == row_index) {
    SL += 1;
  }
  for (int32_t sl = 0; sl < SL; ++sl) {
    int32_t idx = __ldg(&cache_locations[indice_id + sl]);
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<acc_type<float, true>> grad_out_vec(
          &grad_output[row_index * D + d * 4]);
      gpuAtomicAdd(&grad_cache_weight[idx * D + d * 4 + 0], grad_out_vec.acc.x);
      gpuAtomicAdd(&grad_cache_weight[idx * D + d * 4 + 1], grad_out_vec.acc.y);
      gpuAtomicAdd(&grad_cache_weight[idx * D + d * 4 + 2], grad_out_vec.acc.z);
      gpuAtomicAdd(&grad_cache_weight[idx * D + d * 4 + 3], grad_out_vec.acc.w);
    }
  }
}

Tensor cache_backward_dense_cuda(
    int32_t nnz,
    Tensor grad_output,
    Tensor cache_locations,
    Tensor rowidx,
    float learning_rate,
    Tensor cache_weight) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cache_weight.get_device());
  auto grad_cache_weight = zeros_like(cache_weight);
  if (nnz == 0) {
    return grad_cache_weight;
  }

  const auto D = cache_weight.size(1);
  TORCH_CHECK(D > 0);
  TORCH_CHECK(D % 4 == 0);
  int32_t tx = kWarpSize;
  int32_t ty = 1024 / tx;
  dim3 threads(tx, ty);
  int32_t num_blocks = div_round_up(nnz, ty);
  cache_backward_dense_kernel<<<
      num_blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      nnz,
      D,
      grad_output.data_ptr<float>(),
      cache_locations.data_ptr<int32_t>(),
      rowidx.data_ptr<int64_t>(),
      grad_cache_weight.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_cache_weight;
}

__global__ void cache_backward_rowwise_adagrad_approx_kernel(
    int32_t nnz,
    int32_t D,
    const float* __restrict__ grad_output,
    const int32_t* __restrict__ cache_locations,
    const int64_t* __restrict__ rowidx,
    float learning_rate,
    float eps,
    float* __restrict__ cache_optimizer_state,
    float* __restrict__ cache_weight) {
  int32_t indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (indice_id >= nnz) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  // check if this warp is responsible for this whole segment.
  bool segment_start =
      (indice_id == 0 || rowidx[indice_id - 1] != rowidx[indice_id]);
  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x,
    // so we can just exit this warp entirely.
    return;
  }
  int64_t row_index = rowidx[indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (indice_id + SL < nnz && rowidx[indice_id + SL] == row_index) {
    SL += 1;
  }
  float g_local_sum_square = 0.0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<float> grad_out_vec(&grad_output[row_index * D + d * 4]);
    g_local_sum_square += grad_out_vec.acc.x * grad_out_vec.acc.x +
        grad_out_vec.acc.y * grad_out_vec.acc.y +
        grad_out_vec.acc.z * grad_out_vec.acc.z +
        grad_out_vec.acc.w * grad_out_vec.acc.w;
  }
  float g_avg_square = warpReduceAllSum<float>(g_local_sum_square) / D;

  for (int32_t sl = 0; sl < SL; ++sl) {
    auto idx = __ldg(&cache_locations[indice_id + sl]);
    float multiplier;
    if (threadIdx.x == 0) {
      float old_sum_square_grads =
          gpuAtomicAdd(&cache_optimizer_state[idx], g_avg_square);
      multiplier = learning_rate *
          (1.0 / (sqrt(old_sum_square_grads + g_avg_square) + eps));
    }
    multiplier = __shfl_sync(0xFFFFFFFF, multiplier, 0);
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<float> grad_out_vec(&grad_output[row_index * D + d * 4]);
      Vec4T<float> weight_new(&cache_weight[idx * D + d * 4]);
      weight_new.acc.x -= grad_out_vec.acc.x * multiplier;
      weight_new.acc.y -= grad_out_vec.acc.y * multiplier;
      weight_new.acc.z -= grad_out_vec.acc.z * multiplier;
      weight_new.acc.w -= grad_out_vec.acc.w * multiplier;
      weight_new.store(&cache_weight[idx * D + d * 4]);
    }
  }
}

void cache_backward_rowwise_adagrad_approx_cuda(
    int32_t nnz,
    Tensor grad_output,
    Tensor cache_locations,
    Tensor rowidx,
    float learning_rate,
    float eps,
    Tensor cache_optimizer_state,
    Tensor cache_weight) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cache_weight.get_device());
  if (nnz == 0) {
    return;
  }

  const auto D = cache_weight.size(1);
  TORCH_CHECK(D > 0);
  TORCH_CHECK(D % 4 == 0);

  int32_t tx = kWarpSize;
  int32_t ty = 1024 / tx;
  dim3 threads(tx, ty);
  int32_t num_blocks = div_round_up(nnz, ty);
  cache_backward_rowwise_adagrad_approx_kernel<<<
      num_blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      nnz,
      D,
      grad_output.data_ptr<float>(),
      cache_locations.data_ptr<int32_t>(),
      rowidx.data_ptr<int64_t>(),
      learning_rate,
      eps,
      cache_optimizer_state.data_ptr<float>(),
      cache_weight.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
