#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import given, settings, Verbosity
from tt_embeddings_ops import (
    OptimType,
    TableBatchedTTEmbeddingBag,
    tt_matrix_to_full,
    TTEmbeddingBag,
)


def generate_sparse_feature(
    batch_size,
    num_embeddings: int,
    pooling_factor: float,
    pooling_factor_std: float,
    generate_scores: bool = False,
    unary: bool = False,
    unique: bool = False,
) -> Tuple[List, List, List, List]:
    if not unary:
        lengths = np.round(
            np.random.normal(pooling_factor, pooling_factor_std, batch_size)
        ).astype(np.int64)
        lengths = list(np.where(lengths < 0, 0, lengths))
        total_length = np.sum(lengths)
    else:
        lengths = list(np.ones(batch_size).astype(np.int64))
        total_length = batch_size
    indices = list(
        np.random.choice(
            range(num_embeddings), size=total_length, replace=not unique
        ).astype(np.int64)
    )
    if generate_scores:
        scores = list(np.round(np.random.random(total_length) * 20))
    else:
        scores = []
    offsets = [0] + list(np.cumsum(lengths))
    return (lengths, indices, offsets, scores)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class TestTTEmbeddingBag(unittest.TestCase):
    @given(
        batch_size=st.integers(min_value=200, max_value=500),
        pooling_factor=st.integers(min_value=1, max_value=10),
        pooling_factor_std=st.integers(min_value=0, max_value=20),
        tt_ndims=st.integers(min_value=2, max_value=4),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_forward(self, batch_size, pooling_factor, pooling_factor_std, tt_ndims):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        tt_p_shapes = [7, 9, 11, 5]
        tt_q_shapes = [3, 4, 5, 7]
        tt_ranks = [13, 12, 7]
        tt_p_shapes = tt_p_shapes[:tt_ndims]
        tt_q_shapes = tt_q_shapes[:tt_ndims]
        tt_ranks = tt_ranks[: (tt_ndims - 1)]
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))
        _, indices, offsets, _ = generate_sparse_feature(
            batch_size,
            num_embeddings=num_embeddings,
            pooling_factor=float(pooling_factor),
            pooling_factor_std=float(pooling_factor_std),
            generate_scores=False,
            unary=False,
            unique=False,
        )
        # create TT-Embedding op
        offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        indices = torch.tensor(indices, dtype=torch.int64, device=device)
        tt_emb = TTEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_p_shapes=tt_p_shapes,
            tt_q_shapes=tt_q_shapes,
            tt_ranks=tt_ranks,
            sparse=False,
            weight_dist="uniform",
        )
        tt_emb.to(device)
        emb = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            sparse=True,
            mode="sum",
            _weight=tt_emb.full_weight(),
            include_last_offset=True,
        )
        emb.to(device)
        # forward
        output = tt_emb(indices, offsets)
        output_ref = emb(indices.long(), offsets.long())
        torch.testing.assert_allclose(output, output_ref)

    @given(
        batch_size=st.integers(min_value=200, max_value=500),
        pooling_factor=st.integers(min_value=1, max_value=10),
        pooling_factor_std=st.integers(min_value=0, max_value=20),
        tt_ndims=st.integers(min_value=2, max_value=4),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_backward_dense(
        self, batch_size, pooling_factor, pooling_factor_std, tt_ndims
    ):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        tt_p_shapes = [7, 9, 11, 5]
        tt_q_shapes = [3, 4, 5, 7]
        tt_ranks = [13, 12, 7]
        tt_p_shapes = tt_p_shapes[:tt_ndims]
        tt_q_shapes = tt_q_shapes[:tt_ndims]
        tt_ranks = tt_ranks[: (tt_ndims - 1)]
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))
        _, indices, offsets, _ = generate_sparse_feature(
            batch_size,
            num_embeddings=num_embeddings,
            pooling_factor=float(pooling_factor),
            pooling_factor_std=float(pooling_factor_std),
            generate_scores=False,
            unary=False,
            unique=False,
        )
        # create TT-Embedding op
        offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        indices = torch.tensor(indices, dtype=torch.int64, device=device)
        tt_emb = TTEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_p_shapes=tt_p_shapes,
            tt_q_shapes=tt_q_shapes,
            tt_ranks=tt_ranks,
            sparse=False,
            weight_dist="uniform",
        )
        tt_emb.to(device)
        emb = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            sparse=True,
            mode="sum",
            _weight=tt_emb.full_weight(),
            include_last_offset=True,
        )
        emb.to(device)
        d_output = torch.rand(batch_size, embedding_dim, device=device) * 0.1
        tt_cores = [tt.clone().detach().requires_grad_(True) for tt in tt_emb.tt_cores]
        full_weight = tt_matrix_to_full(
            tt_p_shapes, tt_q_shapes, tt_ranks, tt_cores, [1, 0, 2, 3]
        )
        # tt_emb
        output = tt_emb(indices, offsets)
        output.backward(d_output)
        # reference
        output_ref = emb(indices.long(), offsets.long())
        output_ref.backward(d_output)
        d_weight_ref = emb.weight.grad.to_dense()
        full_weight.backward(d_weight_ref)
        for i in range(tt_ndims):
            torch.testing.assert_allclose(tt_emb.tt_cores[i].grad, tt_cores[i].grad)

    @given(
        batch_size=st.integers(min_value=200, max_value=500),
        pooling_factor=st.integers(min_value=1, max_value=10),
        pooling_factor_std=st.integers(min_value=0, max_value=20),
        tt_ndims=st.integers(min_value=2, max_value=4),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_backward_sgd(
        self, batch_size, pooling_factor, pooling_factor_std, tt_ndims
    ):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        tt_p_shapes = [7, 9, 11, 5]
        tt_q_shapes = [3, 4, 5, 7]
        tt_ranks = [13, 12, 7]
        tt_p_shapes = tt_p_shapes[:tt_ndims]
        tt_q_shapes = tt_q_shapes[:tt_ndims]
        tt_ranks = tt_ranks[: (tt_ndims - 1)]
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))
        learning_rate = 0.1
        _, indices, offsets, _ = generate_sparse_feature(
            batch_size,
            num_embeddings=num_embeddings,
            pooling_factor=float(pooling_factor),
            pooling_factor_std=float(pooling_factor_std),
            generate_scores=False,
            unary=False,
            unique=False,
        )
        # create TT-Embedding op
        offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        indices = torch.tensor(indices, dtype=torch.int64, device=device)
        tt_emb = TTEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_p_shapes=tt_p_shapes,
            tt_q_shapes=tt_q_shapes,
            tt_ranks=tt_ranks,
            sparse=True,
            optimizer=OptimType.SGD,
            learning_rate=learning_rate,
            weight_dist="uniform",
        )
        tt_emb.to(device)
        emb = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            sparse=True,
            mode="sum",
            _weight=tt_emb.full_weight(),
            include_last_offset=True,
        )
        emb.to(device)
        d_output = torch.rand(batch_size, embedding_dim, device=device) * 0.1
        tt_cores = [tt.clone().detach().requires_grad_(True) for tt in tt_emb.tt_cores]
        full_weight = tt_matrix_to_full(
            tt_p_shapes, tt_q_shapes, tt_ranks, tt_cores, [1, 0, 2, 3]
        )
        # tt_emb
        output = tt_emb(indices, offsets)
        output.backward(d_output)
        # reference
        output_ref = emb(indices.long(), offsets.long())
        output_ref.backward(d_output)
        d_weight_ref = emb.weight.grad.to_dense()
        full_weight.backward(d_weight_ref)
        new_tt_cores = []
        new_tt_cores = [(t - t.grad * learning_rate) for t in tt_cores]
        for i in range(tt_ndims):
            torch.testing.assert_allclose(tt_emb.tt_cores[i], new_tt_cores[i])

    @given(
        batch_size=st.integers(min_value=200, max_value=500),
        pooling_factor=st.integers(min_value=1, max_value=10),
        pooling_factor_std=st.integers(min_value=0, max_value=20),
        tt_ndims=st.integers(min_value=2, max_value=4),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_backward_adagrad(
        self, batch_size, pooling_factor, pooling_factor_std, tt_ndims
    ):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        tt_p_shapes = [7, 9, 11, 5]
        tt_q_shapes = [3, 4, 5, 7]
        tt_ranks = [13, 12, 7]
        tt_p_shapes = tt_p_shapes[:tt_ndims]
        tt_q_shapes = tt_q_shapes[:tt_ndims]
        tt_ranks = tt_ranks[: (tt_ndims - 1)]
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))
        learning_rate = 0.1
        eps = 0.0001
        _, indices, offsets, _ = generate_sparse_feature(
            batch_size,
            num_embeddings=num_embeddings,
            pooling_factor=float(pooling_factor),
            pooling_factor_std=float(pooling_factor_std),
            generate_scores=False,
            unary=False,
            unique=False,
        )
        # create TT-Embedding op
        offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        indices = torch.tensor(indices, dtype=torch.int64, device=device)
        tt_emb = TTEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_p_shapes=tt_p_shapes,
            tt_q_shapes=tt_q_shapes,
            tt_ranks=tt_ranks,
            sparse=True,
            optimizer=OptimType.EXACT_ADAGRAD,
            learning_rate=learning_rate,
            eps=eps,
            weight_dist="uniform",
        )
        tt_emb.to(device)
        emb = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            sparse=True,
            mode="sum",
            _weight=tt_emb.full_weight(),
            include_last_offset=True,
        )
        emb.to(device)
        d_output = torch.rand(batch_size, embedding_dim, device=device) * 0.1
        tt_cores = [tt.clone().detach().requires_grad_(True) for tt in tt_emb.tt_cores]
        full_weight = tt_matrix_to_full(
            tt_p_shapes, tt_q_shapes, tt_ranks, tt_cores, [1, 0, 2, 3]
        )
        # tt_emb
        output = tt_emb(indices, offsets)
        output.backward(d_output)
        # reference
        output_ref = emb(indices.long(), offsets.long())
        output_ref.backward(d_output)
        d_weight_ref = emb.weight.grad.to_dense()
        full_weight.backward(d_weight_ref)
        new_optimizer_state = []
        new_optimizer_state = [torch.mul(t.grad, t.grad) for t in tt_cores]
        new_tt_cores = []
        new_tt_cores = [
            (
                t
                - torch.div(
                    t.grad * learning_rate, torch.sqrt(new_optimizer_state[i]) + eps
                )
            )
            for i, t in enumerate(tt_cores)
        ]
        for i in range(tt_ndims):
            torch.testing.assert_allclose(
                tt_emb.optimizer_state[i], new_optimizer_state[i]
            )
            torch.testing.assert_allclose(tt_emb.tt_cores[i], new_tt_cores[i])

    @given(
        batch_size=st.integers(min_value=200, max_value=500),
        pooling_factor=st.integers(min_value=1, max_value=10),
        pooling_factor_std=st.integers(min_value=0, max_value=20),
        tt_ndims=st.integers(min_value=2, max_value=4),
        num_tables=st.integers(min_value=1, max_value=4),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_forward_table_batched(
        self, batch_size, pooling_factor, pooling_factor_std, tt_ndims, num_tables
    ):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        tt_p_shapes = [7, 9, 11, 5]
        tt_q_shapes = [3, 4, 5, 7]
        tt_ranks = [13, 12, 7]
        tt_p_shapes = tt_p_shapes[:tt_ndims]
        tt_q_shapes = tt_q_shapes[:tt_ndims]
        tt_ranks = tt_ranks[: (tt_ndims - 1)]
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))

        # create table batched tt embedding bag
        batched_tt_emb = TableBatchedTTEmbeddingBag(
            num_tables=num_tables,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_p_shapes=tt_p_shapes,
            tt_q_shapes=tt_q_shapes,
            tt_ranks=tt_ranks,
            sparse=False,
            weight_dist="uniform",
            use_cache=False,
        )
        batched_tt_emb.to(device)

        tt_embs = []
        lengths_per_table = []
        indices_per_table = []
        inputs_per_table = []
        for i in range(num_tables):
            lengths, indices, offsets, _ = generate_sparse_feature(
                batch_size,
                num_embeddings=num_embeddings,
                pooling_factor=float(pooling_factor),
                pooling_factor_std=float(pooling_factor_std),
                generate_scores=False,
                unary=False,
                unique=False,
            )
            lengths_per_table.extend(lengths)
            indices_per_table.extend(indices)
            offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
            indices = torch.tensor(indices, dtype=torch.int64, device=device)
            inputs_per_table.append((indices, offsets))
            # create TT-Embedding op
            tt_emb = TTEmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                tt_p_shapes=tt_p_shapes,
                tt_q_shapes=tt_q_shapes,
                tt_ranks=tt_ranks,
                sparse=False,
                weight_dist="uniform",
                use_cache=False,
            )
            tt_emb.to(device)
            tt_embs.append(tt_emb)

            # copy tt cores to table batched
            for j, tt_core in enumerate(batched_tt_emb.tt_cores):
                tt_core.detach()[i].copy_(tt_emb.tt_cores[j][0].detach())

        batched_offsets = torch.tensor(
            [0] + list(np.cumsum(lengths_per_table)), dtype=torch.int64, device=device
        )
        batched_indices = torch.tensor(
            indices_per_table, dtype=torch.int64, device=device
        )
        batched_output = batched_tt_emb(batched_indices, batched_offsets)

        assert batched_offsets.numel() - 1 == batch_size * num_tables

        outputs = [
            tt_embs[i](indices, offsets)
            for i, (indices, offsets) in enumerate(inputs_per_table)
        ]

        for i, output in enumerate(outputs):
            # outputs should be close
            torch.testing.assert_allclose(output, batched_output[i])

    @given(
        batch_size=st.integers(min_value=200, max_value=500),
        pooling_factor=st.integers(min_value=1, max_value=10),
        pooling_factor_std=st.integers(min_value=0, max_value=20),
        tt_ndims=st.integers(min_value=2, max_value=4),
        num_tables=st.integers(min_value=1, max_value=4),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_backward_table_batched(
        self, batch_size, pooling_factor, pooling_factor_std, tt_ndims, num_tables
    ):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        tt_p_shapes = [7, 9, 11, 5]
        tt_q_shapes = [3, 4, 5, 7]
        tt_ranks = [13, 12, 7]
        tt_p_shapes = tt_p_shapes[:tt_ndims]
        tt_q_shapes = tt_q_shapes[:tt_ndims]
        tt_ranks = tt_ranks[: (tt_ndims - 1)]
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))

        # create table batched tt embedding bag
        batched_tt_emb = TableBatchedTTEmbeddingBag(
            num_tables=num_tables,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_p_shapes=tt_p_shapes,
            tt_q_shapes=tt_q_shapes,
            tt_ranks=tt_ranks,
            sparse=False,
            weight_dist="uniform",
            use_cache=False,
        )
        batched_tt_emb.to(device)

        tt_embs = []
        lengths_per_table = []
        indices_per_table = []
        inputs_per_table = []
        for i in range(num_tables):
            lengths, indices, offsets, _ = generate_sparse_feature(
                batch_size,
                num_embeddings=num_embeddings,
                pooling_factor=float(pooling_factor),
                pooling_factor_std=float(pooling_factor_std),
                generate_scores=False,
                unary=False,
                unique=False,
            )
            lengths_per_table.extend(lengths)
            indices_per_table.extend(indices)
            offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
            indices = torch.tensor(indices, dtype=torch.int64, device=device)
            inputs_per_table.append((indices, offsets))
            # create TT-Embedding op
            tt_emb = TTEmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                tt_p_shapes=tt_p_shapes,
                tt_q_shapes=tt_q_shapes,
                tt_ranks=tt_ranks,
                sparse=False,
                weight_dist="uniform",
                use_cache=False,
            )
            tt_emb.to(device)
            tt_embs.append(tt_emb)

            # copy tt cores to table batched
            for j, tt_core in enumerate(batched_tt_emb.tt_cores):
                tt_core.detach()[i].copy_(tt_emb.tt_cores[j][0].detach())

        batched_offsets = torch.tensor(
            [0] + list(np.cumsum(lengths_per_table)), dtype=torch.int64, device=device
        )
        batched_indices = torch.tensor(
            indices_per_table, dtype=torch.int64, device=device
        )
        batched_output = batched_tt_emb(batched_indices, batched_offsets)

        assert batched_offsets.numel() - 1 == batch_size * num_tables

        outputs = [
            tt_embs[i](indices, offsets)
            for i, (indices, offsets) in enumerate(inputs_per_table)
        ]

        d_batched_output = (
            torch.rand(num_tables, batch_size, embedding_dim, device=device) * 0.1
        )

        batched_output.backward(d_batched_output)
        for i, output in enumerate(outputs):
            output.backward(d_batched_output[i])
            for j, tt_core in enumerate(tt_embs[i].tt_cores):
                torch.testing.assert_allclose(
                    tt_core.grad[0], batched_tt_emb.tt_cores[j].grad[i]
                )
