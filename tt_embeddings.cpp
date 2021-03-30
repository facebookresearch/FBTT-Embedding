/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/

#include <ATen/ATen.h>
#include <torch/extension.h>

using namespace at;

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
    Tensor indices,
    Tensor rowidx,
    Tensor tableidx,
    const std::vector<Tensor>& tt_cores);

std::vector<Tensor> tt_embeddings_backward_dense_cuda(
    int32_t batch_count,
    int32_t D,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor indices,
    Tensor offsets,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores);

void tt_embeddings_backward_sgd_cuda(
    int32_t batch_count,
    int32_t D,
    float learning_rate,
    const std::vector<int32_t>& tt_p_shapes,
    const std::vector<int32_t>& tt_q_shapes,
    const std::vector<int32_t>& tt_ranks,
    Tensor L,
    int32_t nnz,
    Tensor indices,
    Tensor offsets,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& tt_cores);

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
    Tensor indices,
    Tensor offsets,
    Tensor tableidx,
    Tensor d_output,
    std::vector<Tensor>& optimizer_state,
    std::vector<Tensor>& tt_cores);

void update_cache_state_cuda(Tensor indices, Tensor hashtbl, Tensor cache_freq);

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
    Tensor cache_weight);

std::tuple<Tensor, Tensor, Tensor, int32_t, c10::optional<Tensor>>
preprocess_indices_sync_cuda(
    Tensor colidx,
    Tensor offsets,
    int32_t num_tables,
    bool warmup,
    Tensor hashtbl,
    Tensor cache_state);

void cache_forward_cuda(
    int32_t B,
    int32_t nnz,
    Tensor cache_locations,
    Tensor rowidx,
    Tensor cache_weight,
    Tensor output);

void cache_backward_sgd_cuda(
    int32_t nnz,
    Tensor grad_output,
    Tensor cache_locations,
    Tensor rowidx,
    float learning_rate,
    Tensor cache_weight);

Tensor cache_backward_dense_cuda(
    int32_t nnz,
    Tensor grad_output,
    Tensor cache_locations,
    Tensor rowidx,
    float learning_rate,
    Tensor cache_weight);

void cache_backward_rowwise_adagrad_approx_cuda(
    int32_t nnz,
    Tensor grad_output,
    Tensor cache_locations,
    Tensor rowidx,
    float learning_rate,
    float eps,
    Tensor cache_optimizer_state,
    Tensor cache_weight);

PYBIND11_MODULE(tt_embeddings, m) {
  m.def("tt_forward", &tt_embeddings_forward_cuda, "tt_forward()");
  m.def(
      "tt_dense_backward",
      &tt_embeddings_backward_dense_cuda,
      "tt_dense_backward()");
  m.def(
      "tt_sgd_backward", &tt_embeddings_backward_sgd_cuda, "tt_sgd_backward()");
  m.def(
      "tt_adagrad_backward",
      &tt_embeddings_backward_adagrad_cuda,
      "tt_adagrad_backward()");

  m.def("update_cache_state", &update_cache_state_cuda, "update_cache_state()");
  m.def("cache_populate", &cache_populate_cuda, "cache_populate()");
  m.def(
      "preprocess_indices_sync",
      &preprocess_indices_sync_cuda,
      "preprocess_colidx_sync()");

  m.def("cache_forward", &cache_forward_cuda, "cache_forward()");
  m.def("cache_backward_sgd", &cache_backward_sgd_cuda, "cache_backward_sgd()");
  m.def(
      "cache_backward_dense",
      &cache_backward_dense_cuda,
      "cache_backward_dense()");
  m.def(
      "cache_backward_rowwise_adagrad_approx",
      &cache_backward_rowwise_adagrad_approx_cuda,
      "cache_backward_rowwise_adagrad_approx()");
}
