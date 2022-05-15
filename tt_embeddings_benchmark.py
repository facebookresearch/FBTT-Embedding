#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, List, Optional, Tuple

import click
import numpy as np
import torch
from tt_embeddings_ops import OptimType, TTEmbeddingBag


logging.basicConfig(level=logging.DEBUG)


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.contiguous().view(-1),
        torch.tensor(
            ([0] + np.cumsum(flat_lengths).tolist()),
            device=torch.cuda.current_device(),
            dtype=merged_indices.dtype,
        ),
    )


def generate_requests(
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    long_index: bool = True,
    # inter-batch indices reuse rate
    reuse: float = 0.0,
    # alpha <= 1.0: use uniform distribution
    # alpha > 1.0: use zjpf distribution
    alpha: float = 1.0,
    fp16: bool = False,
    weighted: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    index_type = torch.int64 if long_index else torch.int32
    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B * L),
            device=torch.cuda.current_device(),
            dtype=index_type,
        )
    else:
        all_indices = (
            torch.as_tensor(
                np.random.zipf(a=alpha, size=(iters, T, B * L)),
                device=torch.cuda.current_device(),
                dtype=index_type,
            )
            % E
        )
    for it in range(iters - 1):
        for t in range(T):
            reused_indices = torch.randperm(B * L, device=torch.cuda.current_device())[
                : int(B * L * reuse)
            ]
            all_indices[it + 1, t, reused_indices] = all_indices[it, t, reused_indices]

    rs = [
        get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L))
        + (
            torch.randn(
                T * B * L,
                device=torch.cuda.current_device(),
                dtype=torch.float16 if fp16 else torch.float32,
            )
            if weighted
            else None,
        )
        for it in range(iters)
    ]
    # pyre-fixme[7]
    return rs


def benchmark_requests(
    requests: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    f: Callable,
):
    for (indices, offsets, weights) in requests:
        f(indices, offsets, weights)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for (indices, offsets, weights) in requests:
        f(indices, offsets, weights)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / len(requests)


def validate_list(ctx: click.core.Context, param: click.core.Option, param_str: str):
    values = []
    try:
        for v in param_str.strip().split(","):
            if int(v) <= 0:
                raise click.BadParameter(f"Invalid parameter '{param_str}'")
            values.append(int(v))
    except ValueError:
        raise click.BadParameter(f"Invalid parameter '{param_str}'")
    return values


@click.command()
@click.option("--batch-size", default=512)
@click.option("--iters", default=10)
@click.option("--pooling-factor", default=20)
@click.option("--p-shapes", default="200,220,250", callback=validate_list)
@click.option("--q-shapes", default="4,4,4", callback=validate_list)
@click.option("--ranks", default="32,32", callback=validate_list)
@click.option("--long-index", is_flag=True, default=True)
@click.option("--sparse", is_flag=True, default=True)
@click.option("--optimizer", default="sgd")
@click.option("--run-baseline", is_flag=True, default=False)
def main(
    batch_size,
    iters,
    long_index,
    pooling_factor,
    p_shapes,
    q_shapes,
    ranks,
    sparse,
    optimizer,
    run_baseline,
):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    num_embeddings = np.prod(np.array(p_shapes))
    embedding_dim = np.prod(np.array(q_shapes))
    requests = generate_requests(
        iters, batch_size, 1, pooling_factor, num_embeddings, long_index
    )
    nnz = batch_size * pooling_factor
    flop = (
        q_shapes[0] * ranks[0] * q_shapes[1] * ranks[1]
        + q_shapes[0] * q_shapes[1] * ranks[1] * q_shapes[2]
    )
    flop = 2.0 * nnz * flop * iters
    bw = 4.0 * nnz * embedding_dim * iters

    # create TT-Embedding op
    if optimizer == "sgd":
        optimizer = OptimType.SGD
    else:
        optimizer = OptimType.EXACT_ADAGRAD
    tt_emb = TTEmbeddingBag(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        tt_p_shapes=p_shapes,
        tt_q_shapes=q_shapes,
        tt_ranks=ranks,
        sparse=sparse,
        optimizer=optimizer,
        use_cache=True,
    )
    tt_emb.to(device)
    logging.info(f"sparse: {sparse}, optimizer: {optimizer}")
    logging.info(f"p_shapes: {p_shapes}, " f"q_shapes: {q_shapes}, " f"ranks: {ranks}")
    logging.info(
        f"B: {batch_size}, E: {num_embeddings}, " f"D: {embedding_dim}, nnz: {nnz}"
    )

    grad_output = torch.rand(batch_size, embedding_dim, device=device) * 0.1
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: tt_emb(indices, offsets).backward(grad_output),
    )
    logging.info(
        f"TTEmbeddingBag FWD-BWD time/nnz: {time_per_iter / nnz * 1e6: .3f} usecs, "
        f"GFLOPS: {3.0 * flop / time_per_iter / 1e9: .3f}, "
        f"BW: {3.0 * bw / time_per_iter / 1e9: .3f}"
    )

    # EmbeddingBag
    if run_baseline:
        emb = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            sparse=True,
            mode="sum",
            include_last_offset=True,
        )
        emb.to(device)
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, _: emb(indices, offsets).backward(grad_output),
        )
        logging.info(
            f"EmbeddingBag FWD-BWD time/nnz: {time_per_iter / nnz * 1e6: .3f} usecs, "
            f"BW: {3.0 * bw / time_per_iter / 1e9: .3f}"
        )


if __name__ == "__main__":
    main()
