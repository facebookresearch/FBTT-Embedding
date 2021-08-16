#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tt_embeddings
from torch import nn


@unique
class OptimType(Enum):
    SGD = "sgd"  # uses non-deterministic updates (atomicAdd(..)) with duplicate ids
    EXACT_SGD = (
        "exact_sgd"  # uses deterministic updates (via sorting + segment reduction)
    )
    LAMB = "lamb"
    ADAM = "adam"
    EXACT_ADAGRAD = "exact_adagrad"
    EXACT_ROWWISE_ADAGRAD = "exact_row_wise_adagrad"
    LARS_SGD = "lars_sgd"
    PARTIAL_ROWWISE_ADAM = "partial_row_wise_adam"
    PARTIAL_ROWWISE_LAMB = "partial_row_wise_lamb"

    def __str__(self):
        return self.value


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, name: str, buffers: Optional[List[torch.Tensor]] = None):
        super(BufferList, self).__init__()
        self._name = name
        self._offset = 0
        if buffers is not None:
            self.extend(buffers)
            self._length = len(buffers)
        else:
            self._length = 0

    def extend(self, buffers: List[torch.Tensor]) -> "BufferList":
        for i, buffer in enumerate(buffers):
            self.register_buffer(self._name + str(self._length + i), buffer)
        self._length += len(buffers)
        return self

    def append(self, buffer: torch.Tensor) -> "BufferList":
        self.register_buffer(self._name + str(self._length), buffer)
        self._length += 1
        return self

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        self._offset = 0
        return self

    def __next__(self):
        if self._offset < self._length:
            self._offset += 1
            return getattr(self, self._name + str(self._offset - 1))
        else:
            raise StopIteration

    def __getitem__(self, index: int) -> torch.Tensor:
        return getattr(self, self._name + str(index))


def tt_matrix_to_full(
    tt_p_shapes: List[int],
    tt_q_shapes: List[int],
    tt_ranks: List[int],
    tt_cores: List[torch.Tensor],
    tt_permute: Optional[List[int]] = None,
) -> torch.Tensor:
    tt_ndim = len(tt_p_shapes)
    if len(tt_ranks) == tt_ndim - 1:
        tt_ranks = [1] + tt_ranks + [1]
    tt_cores_ = []
    if tt_permute is not None:
        for i, t in enumerate(tt_cores):
            size_tt = [tt_ranks[i], tt_p_shapes[i], tt_q_shapes[i], tt_ranks[i + 1]]
            size_tt_permute = [0] * 4
            for i in range(4):
                size_tt_permute[i] = size_tt[tt_permute[i]]
            tt_cores_.append(t.view(*size_tt_permute).permute(*tt_permute).contiguous())
    else:
        for t in tt_cores:
            tt_cores_.append(torch.squeeze(t))
    for k in range(tt_ndim):
        assert tt_cores_[k].size(0) == tt_ranks[k]
        assert tt_cores_[k].size(1) == tt_p_shapes[k]
        assert tt_cores_[k].size(2) == tt_q_shapes[k]
        assert tt_cores_[k].size(3) == tt_ranks[k + 1]
    res = tt_cores_[0]
    for i in range(1, tt_ndim):
        res = res.view(-1, tt_ranks[i])
        curr_core = tt_cores_[i].view(tt_ranks[i], -1)
        res = torch.matmul(res, curr_core)
    intermediate_shape = []
    n_dim = 1
    k_dim = 1
    for i in range(tt_ndim):
        intermediate_shape.append(tt_p_shapes[i])
        intermediate_shape.append(tt_q_shapes[i])
        n_dim *= tt_p_shapes[i]
        k_dim *= tt_q_shapes[i]
    res = res.view(*intermediate_shape)
    transpose = []
    for i in range(0, 2 * tt_ndim, 2):
        transpose.append(i)
    for i in range(1, 2 * tt_ndim, 2):
        transpose.append(i)
    res = res.permute(*transpose)
    res = res.contiguous().view(n_dim, k_dim).float()
    return res


class TTLookupFunction(torch.autograd.Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        B: int,
        D: int,
        tt_p_shapes: List[int],
        tt_q_shapes: List[int],
        tt_ranks: List[int],
        L: torch.Tensor,
        nnz_tt: int,
        nnz_cached: int,
        indices: torch.Tensor,
        rowidx: torch.Tensor,
        tableidx: torch.Tensor,
        optimizer: OptimType,
        learning_rate: float,
        eps: float,
        sparse: bool,
        cache_locations: torch.Tensor,
        cache_optimizer_state: torch.Tensor,
        cache_weight: torch.Tensor,
        optimizer_state: List[torch.Tensor],
        *tt_cores: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        ctx.tt_p_shapes = tt_p_shapes
        ctx.tt_q_shapes = tt_q_shapes
        ctx.tt_ranks = tt_ranks
        ctx.D = D
        ctx.optimizer = optimizer
        ctx.learning_rate = learning_rate
        ctx.eps = eps
        ctx.sparse = sparse
        ctx.tt_cores = tt_cores
        ctx.optimizer_state = optimizer_state
        ctx.nnz_tt = nnz_tt
        ctx.nnz_cached = nnz_cached
        batch_count = 1000
        ctx.save_for_backward(
            L,
            indices,
            rowidx,
            tableidx,
            cache_locations,
            cache_optimizer_state,
            cache_weight,
        )
        # pyre-fixme[16]
        output = tt_embeddings.tt_forward(
            batch_count,
            ctx.tt_cores[0].size(0),  # num_tables
            B,
            D,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            L,
            nnz_tt,
            indices,
            rowidx,
            tableidx,
            list(ctx.tt_cores),
        )
        if nnz_cached > 0:
            # pyre-fixme[16]
            tt_embeddings.cache_forward(
                B,
                nnz_cached,
                cache_locations[ctx.nnz_tt :],
                rowidx[nnz_tt:],
                cache_weight,
                output,
            )

        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(ctx, d_output: torch.Tensor) -> Tuple[torch.Tensor]:
        (
            L,
            indices,
            rowidx,
            tableidx,
            cache_locations,
            cache_optimizer_state,
            cache_weight,
        ) = ctx.saved_tensors
        batch_count = 1000
        if ctx.sparse:
            if ctx.optimizer in [OptimType.SGD, OptimType.EXACT_SGD]:
                # pyre-fixme[16]
                tt_embeddings.tt_sgd_backward(
                    batch_count,
                    ctx.D,
                    ctx.learning_rate,
                    ctx.tt_p_shapes,
                    ctx.tt_q_shapes,
                    ctx.tt_ranks,
                    L,
                    ctx.nnz_tt,
                    indices,
                    rowidx,
                    tableidx,
                    d_output,
                    list(ctx.tt_cores),
                )
                if ctx.nnz_cached > 0:
                    # pyre-fixme[16]
                    tt_embeddings.cache_backward_sgd(
                        ctx.nnz_cached,
                        d_output,
                        cache_locations[ctx.nnz_tt :],
                        rowidx[ctx.nnz_tt :],
                        ctx.learning_rate,
                        cache_weight,
                    )
            else:
                # pyre-fixme[16]
                tt_embeddings.tt_adagrad_backward(
                    batch_count,
                    ctx.D,
                    ctx.learning_rate,
                    ctx.eps,
                    ctx.tt_p_shapes,
                    ctx.tt_q_shapes,
                    ctx.tt_ranks,
                    L,
                    ctx.nnz_tt,
                    indices,
                    rowidx,
                    tableidx,
                    d_output,
                    ctx.optimizer_state,
                    list(ctx.tt_cores),
                )
                if ctx.nnz_cached > 0:
                    # pyre-fixme[16]
                    tt_embeddings.cache_backward_rowwise_adagrad_approx(
                        ctx.nnz_cached,
                        d_output,
                        cache_locations[ctx.nnz_tt :],
                        rowidx[ctx.nnz_tt :],
                        ctx.learning_rate,
                        ctx.eps,
                        cache_optimizer_state,
                        cache_weight,
                    )
            # pyre-fixme[7]
            return tuple(
                [
                    None,  # D
                    None,  # tt_p_shapes
                    None,  # tt_q_shapes
                    None,  # tt_ranks
                    None,  # K
                    None,  # nnz_tt
                    None,  # nnz_cached
                    None,  # indices
                    None,  # offsets
                    None,  # rowidx
                    None,  # tableidx
                    None,  # optimizer
                    None,  # learning_rate
                    None,  # eps
                    None,  # sparse
                    None,  # cache_locations
                    None,  # cache_optimizer_state
                    None,  # cache_weight
                    None,  # optimizer_state
                ]
                + [None] * len(ctx.tt_cores)
            )
        else:
            # pyre-fixme[16]
            d_tt_cores = tt_embeddings.tt_dense_backward(
                batch_count,
                ctx.D,
                ctx.tt_p_shapes,
                ctx.tt_q_shapes,
                ctx.tt_ranks,
                L,
                ctx.nnz_tt,
                indices,
                rowidx,
                tableidx,
                d_output,
                list(ctx.tt_cores),
            )
            if ctx.nnz_cached > 0:
                # pyre-fixme[16]
                d_cache_weight = tt_embeddings.cache_backward_dense(
                    ctx.nnz_cached,
                    d_output,
                    cache_locations[ctx.nnz_tt :],
                    rowidx[ctx.nnz_tt :],
                    ctx.learning_rate,
                    cache_weight,
                )
            else:
                d_cache_weight = None
            # pyre-fixme[7]
            return tuple(
                [
                    None,  # D
                    None,  # tt_p_shapes
                    None,  # tt_q_shapes
                    None,  # tt_ranks
                    None,  # K
                    None,  # nnz_tt
                    None,  # nnz_cached
                    None,  # indices
                    None,  # offsets
                    None,  # rowidx
                    None,  # tableidx
                    None,  # optimizer
                    None,  # learning_rate
                    None,  # eps
                    None,  # sparse
                    None,  # cache_locations
                    None,  # cache_optimizer_state
                    d_cache_weight,  # cache_weight
                    None,  # optimizer_state
                ]
                + d_tt_cores
            )


def suggested_tt_shapes(  # noqa C901
    n: int, d: int = 3, allow_round_up: bool = True
) -> List[int]:
    from itertools import cycle, islice

    # pyre-fixme[21]
    from scipy.stats import entropy
    from sympy.ntheory import factorint
    from sympy.utilities.iterables import multiset_partitions

    def _auto_shape(n: int, d: int = 3) -> List[int]:
        def _to_list(x: Dict[int, int]) -> List[int]:
            res = []
            for k, v in x.items():
                res += [k] * v
            return res

        p = _to_list(factorint(n))
        if len(p) < d:
            p = p + [1] * (d - len(p))

        def _roundrobin(*iterables):
            pending = len(iterables)
            nexts = cycle(iter(it).__next__ for it in iterables)
            while pending:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    pending -= 1
                    nexts = cycle(islice(nexts, pending))

        def prepr(x: List[int]) -> Tuple:
            x = sorted(np.prod(_) for _ in x)
            N = len(x)
            xf, xl = x[: N // 2], x[N // 2 :]
            return tuple(_roundrobin(xf, xl))

        raw_factors = multiset_partitions(p, d)
        clean_factors = [prepr(f) for f in raw_factors]
        factors = list(set(clean_factors))
        # pyre-fixme[16]
        weights = [entropy(f) for f in factors]
        i = np.argmax(weights)
        return list(factors[i])

    def _roundup(n: int, k: int) -> int:
        return int(np.ceil(n / 10 ** k)) * 10 ** k

    if allow_round_up:
        weights = []
        for i in range(len(str(n))):
            n_i = _roundup(n, i)
            # pyre-fixme[16]
            weights.append(entropy(_auto_shape(n_i, d=d)))
        i = np.argmax(weights)
        factors = _auto_shape(_roundup(n, i), d=d)
    else:
        factors = _auto_shape(n, d=d)
    return factors


class TableBatchedTTEmbeddingBag(torch.nn.Module):
    """
    TT embedding bag that supports looking up multiple tables in one pass.
    It has to satisfy the constraint that all tables have the same num_embeddings and embedding_dim
    """

    __constants__ = [
        "num_tables",
        "num_embeddings",
        "embedding_dim",
        "tt_shape",
        "tt_rank",
    ]

    def __init__(
        self,
        num_tables: int,
        num_embeddings: int,
        embedding_dim: int,
        tt_ranks: List[int],
        tt_p_shapes: Optional[List[int]] = None,
        tt_q_shapes: Optional[List[int]] = None,
        optimizer: OptimType = OptimType.SGD,
        learning_rate: float = 0.1,
        eps: float = 1.0e-10,
        sparse: bool = True,
        use_cache: bool = False,
        cache_size: int = 0,
        hashtbl_size: int = 0,
        weight_dist: str = "approx-normal",
        enforce_embedding_dim: bool = False,
    ) -> None:
        super(TableBatchedTTEmbeddingBag, self).__init__()
        assert torch.cuda.is_available()
        assert num_tables > 0
        assert num_embeddings > 0
        assert embedding_dim > 0
        assert num_tables == 1 or not use_cache, "cannot use cache when num_tables != 1"
        self.tt_p_shapes: List[int] = (
            suggested_tt_shapes(num_embeddings, len(tt_ranks) + 1)
            if tt_p_shapes is None
            else tt_p_shapes
        )
        self.tt_q_shapes: List[int] = (
            # if enforce_embedding_dim=True, we make sure that
            # prod(tt_q_shapes) == embedding_dim by disabling round up
            suggested_tt_shapes(
                embedding_dim,
                len(tt_ranks) + 1,
                allow_round_up=(not enforce_embedding_dim),
            )
            if tt_q_shapes is None
            else tt_q_shapes
        )
        assert len(self.tt_p_shapes) >= 2
        assert len(self.tt_p_shapes) <= 4
        assert len(tt_ranks) + 1 == len(self.tt_p_shapes)
        assert len(self.tt_p_shapes) == len(self.tt_q_shapes)
        assert all(v > 0 for v in self.tt_p_shapes)
        assert all(v > 0 for v in self.tt_q_shapes)
        assert all(v > 0 for v in tt_ranks)
        assert np.prod(np.array(self.tt_p_shapes)) >= num_embeddings
        assert np.prod(np.array(self.tt_q_shapes)) == embedding_dim
        self.num_tables = num_tables
        self.tt_ndim = len(tt_ranks) + 1
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tt_ranks = [1] + tt_ranks + [1]
        self.sparse = sparse
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.eps = eps
        logging.info(
            f"Creating TTEmbeddingBag "
            f"tt_p_shapes: {self.tt_p_shapes}, "
            f"tt_q_shapes: {self.tt_q_shapes}, "
            f"tt_ranks: {self.tt_ranks}, "
            f"sparse: {self.sparse}, "
            f"optimizer: {self.optimizer}, "
            f"learning_rate: {self.learning_rate}, "
            f"eps: {self.eps}"
            f"use_cache: {use_cache}, "
            f"cache_size: {cache_size}, "
            f"hashtbl_size: {hashtbl_size}"
        )
        L = []
        L_value = 1
        for t in range(self.tt_ndim):
            L.append(L_value)
            L_value *= self.tt_p_shapes[self.tt_ndim - t - 1]
        L.reverse()
        self.register_buffer("L", torch.tensor(L, dtype=torch.int64))
        self.tt_cores = torch.nn.ParameterList()
        self.optimizer_state = BufferList("optimizer_state")
        for i in range(self.tt_ndim):
            self.tt_cores.append(
                torch.nn.Parameter(
                    torch.empty(
                        [
                            self.num_tables,
                            self.tt_p_shapes[i],
                            self.tt_ranks[i]
                            * self.tt_q_shapes[i]
                            * self.tt_ranks[i + 1],
                        ],
                        device=torch.cuda.current_device(),
                        dtype=torch.float32,
                    )
                )
            )
            optimizer_state_shape = (
                self.tt_cores[i].shape
                if self.optimizer not in [OptimType.SGD, OptimType.EXACT_SGD]
                else 0
            )
            self.optimizer_state.append(
                torch.zeros(
                    optimizer_state_shape,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
        self.reset_parameters(weight_dist)
        self.use_cache = use_cache
        if use_cache:
            if cache_size <= 0:
                cache_size = int(0.1 * self.num_embeddings)
            if hashtbl_size <= 0:
                hashtbl_size = self.num_embeddings
            assert hashtbl_size >= cache_size
            self.register_buffer(
                "hashtbl",
                torch.empty(
                    hashtbl_size, device=torch.cuda.current_device(), dtype=torch.int64
                ).fill_(-1),
            )
            self.register_buffer(
                "cache_freq",
                torch.zeros(
                    hashtbl_size, device=torch.cuda.current_device(), dtype=torch.int64
                ),
            )
            self.register_buffer(
                "cache_state",
                torch.empty(
                    hashtbl_size, device=torch.cuda.current_device(), dtype=torch.int32
                ).fill_(-1),
            )
            self.cache_weight = nn.Parameter(
                torch.zeros(
                    (cache_size, self.embedding_dim),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
            if self.sparse and optimizer not in (OptimType.SGD, OptimType.EXACT_SGD):
                optimizer_state_shape = (
                    (cache_size, self.embedding_dim)
                    if optimizer == OptimType.EXACT_ADAGRAD
                    else (cache_size)
                )
                self.register_buffer(
                    "cache_optimizer_state",
                    torch.zeros(optimizer_state_shape, dtype=torch.float32),
                )
            else:
                self.cache_optimizer_state = None
        else:
            self.register_buffer(
                "hashtbl",
                torch.empty(0, device=torch.cuda.current_device(), dtype=torch.int64),
            )
            self.register_buffer(
                "cache_state",
                torch.empty(0, device=torch.cuda.current_device(), dtype=torch.int32),
            )
            self.cache_optimizer_state = None
            self.cache_weight = None
        self.warmup = True

    def full_weight(self) -> torch.Tensor:
        assert (
            self.num_tables == 1
        ), "full_weight() only supported for num_tables == 1 for now"
        return tt_matrix_to_full(
            self.tt_p_shapes,
            self.tt_q_shapes,
            self.tt_ranks,
            self.tt_cores,
            [1, 0, 2, 3],
        )

    def reset_parameters(self, weight_dist: str) -> None:  # noqa C901
        assert weight_dist in [
            "uniform",
            "naive-uniform",
            "normal",
            "approx-uniform",
            "approx-normal",
        ]
        if weight_dist == "uniform":
            lamb = 2.0 / (self.num_embeddings + self.embedding_dim)
            stddev = np.sqrt(lamb)
            tt_ranks = np.array(self.tt_ranks)
            cr_exponent = -1.0 / (2 * self.tt_ndim)
            var = np.prod(tt_ranks ** cr_exponent)
            core_stddev = stddev ** (1.0 / self.tt_ndim) * var
            for i in range(self.tt_ndim):
                torch.nn.init.uniform_(self.tt_cores[i], 0.0, core_stddev)
        elif weight_dist == "naive-uniform":
            for i in range(self.tt_ndim):
                torch.nn.init.uniform_(
                    self.tt_cores[i], 0.0, 1 / np.sqrt(self.num_embeddings)
                )
        elif weight_dist == "normal":
            mu = 0.0
            sigma = 1.0 / np.sqrt(self.num_embeddings)
            scale = 1.0 / self.tt_ranks[0]
            for i in range(self.tt_ndim):
                torch.nn.init.normal_(self.tt_cores[i], mu, sigma)
                self.tt_cores[i].data *= scale
        elif weight_dist == "approx-normal":
            mu = 0.0
            sigma = 1.0
            scale = np.power(1 / np.sqrt(3 * self.num_embeddings), 1 / 3)
            for i in range(self.tt_ndim):
                W = np.random.normal(
                    loc=mu, scale=sigma, size=np.asarray(self.tt_cores[i].shape)
                ).astype(np.float32)
                core_shape = self.tt_cores[i].shape
                W = W.flatten()
                for ele in range(W.shape[0]):
                    while np.abs(W[ele]) < 2:
                        W[ele] = np.random.normal(loc=mu, scale=sigma, size=[1]).astype(
                            np.float32
                        )
                W = np.reshape(W, core_shape)
                W *= scale
                self.tt_cores[i].data = torch.tensor(W, requires_grad=True)
        elif weight_dist == "approx-uniform":

            def _flat_saw_tooth(nb_gridpts: int, width: float, nb_samples: int = 1):
                """
                This is a "flat saw tooth" distribution
                that is, the density function is a sum of
                j*delta + uniform(-width/2, width/2), width < delta/2 in general
                a finite train of flat tooth with space in between
                The idea is that when this density function convolved
                with a very narrow gaussian-like distribution
                the space will be filled up and the result looks like a uniform distribiution
                """

                N = nb_gridpts
                delta = 1.0 / N
                j = np.random.randint(-(N - 1), N, nb_samples)
                x = -width / 2.0 + width * np.random.rand(nb_samples)
                return j * delta + x

            def _gen_block(
                dist: str, dim: List[int], center: float, param: float
            ) -> np.ndarray:
                nb_samples = (np.array(dim)).prod()
                if dist == "gaussian":
                    B = center + np.random.randn(nb_samples) * param
                elif dist == "uniform":
                    B = center - (param / 2.0) + param * np.random.rand(nb_samples)
                else:
                    assert 0, f"Does not support {dist} distribution"
                # pyre-fixme[16]
                B = B.reshape(dim)
                return B

            def _gen_head(dim: List[int], sigma: float = 0.01) -> np.ndarray:
                # expect dim = (1, m1, n1, r1) where r1 is the tensor train rank
                scale = 1.0 / np.sqrt(dim[-1])
                size = (np.array(dim)).prod()
                B = _gen_block("gaussian", size, scale, sigma)
                B = B.reshape(dim)
                return B

            def _gen_tail(
                dim: List[int],
                sigma: float = 0.01,
                nb_gridpts: int = 15,
                width: float = 0.7 / 30.0,
            ):
                """
                expect dim = (r3, m3, n3, 1); r3 is the tensor train rank
                in our scheme here, all the elements are small, N(0,sigma^2)
                except on each possible m, n  there is one random odd r
                such that (r, m, n, 1) follows a saw tooth distribution
                """
                # first generate all the backgrounds as one big block
                B = _gen_block("gaussian", dim, 0.0, sigma)
                # generate the needed saw tooth distribution
                r3 = dim[0]
                B = B.reshape(r3, -1)
                nb_samples = B.shape[1]
                values = _flat_saw_tooth(nb_gridpts, width, nb_samples=nb_samples)
                for ell in range(nb_samples):
                    p = random.randrange(1, r3, 2)
                    B[p, ell] = values[ell]
                B = B.reshape(dim)
                return B

            def _gen_mid(
                dim: List[int],
                sigma: float = 0.01,
                nb_gridpts: int = 15,
                width: float = 0.7 / 30.0,
            ):
                """
                expect dim = (r2, m2, n2, r3)
                in our scheme, all the elements are in general close to 1/sqrt(r2)
                so that the product with the head yield
                values close to 1
                but for each specific value of (m,n) in the range of (m2,n2)
                we pick a random even index k in range of r3 such that we
                make the vector (:,m,n,k) to be small except
                for one random j in range of r2 so that the value (j,m,n,k)
                is drawn for a saw tooth distribution
                so the total number of needed saw tooth samples is m2 x n2
                """
                r2, m2, n2, r3 = dim
                scale = 1.0 / np.sqrt(r2)
                B = _gen_block("gaussian", dim, scale, sigma)
                B = B.reshape(r2, m2 * n2, r3)
                values = _flat_saw_tooth(nb_gridpts, width, nb_samples=m2 * n2) / scale
                for ell in range(m2 * n2):
                    p = random.randrange(0, r3, 2)
                    v = np.random.randn(r2) * (sigma * sigma / scale)
                    B[:, ell, p] = v
                    j = random.randrange(r2)
                    B[j, ell, p] = values[ell]
                B = B.reshape(dim)
                return B

            assert self.tt_ndim == 3
            assert (
                self.num_tables == 1
            ), "approx_uniform only supported for num_tables == 1"
            scale = 1.0 / (np.sqrt(self.num_embeddings) ** (1.0 / 3.0))
            shapes = []
            for i in range(self.tt_ndim):
                core_shape = [
                    self.tt_ranks[i],
                    self.tt_p_shapes[i],
                    self.tt_q_shapes[i],
                    self.tt_ranks[i + 1],
                ]
                shapes.append(core_shape)
            W0 = _gen_head(shapes[0], sigma=0.01)
            W0 = W0 * scale
            W0 = W0.transpose([1, 0, 2, 3]).reshape(
                (self.num_tables, self.tt_p_shapes[0], -1)
            )
            W0 = W0.astype(np.float32)
            W1 = _gen_mid(shapes[1], sigma=0.01)
            W1 = W1 * scale
            W1 = W1.astype(np.float32)
            W1 = W1.transpose([1, 0, 2, 3]).reshape(
                (self.num_tables, self.tt_p_shapes[1], -1)
            )
            W2 = _gen_tail(shapes[2], sigma=0.01)
            W2 = W2 * scale
            W2 = W2.astype(np.float32)
            W2 = W2.transpose([1, 0, 2, 3]).reshape(
                (self.num_tables, self.tt_p_shapes[2], -1)
            )
            self.tt_cores[0].data = torch.tensor(W0, requires_grad=True)
            self.tt_cores[1].data = torch.tensor(W1, requires_grad=True)
            self.tt_cores[2].data = torch.tensor(W2, requires_grad=True)

    def reset_cache(self):
        if self.use_cahce:
            self.hashtbl.fill_(-1)
            self.cache_freq.fill_(0)
            self.cache_state.fill_(-1)

    def cache_populate(self):
        if self.use_cache:
            tt_embeddings.cache_populate(
                self.num_embeddings,
                self.tt_p_shapes,
                self.tt_q_shapes,
                self.tt_ranks,
                self.tt_cores,
                self.L,
                self.hashtbl,
                self.cache_freq,
                self.cache_state,
                self.cache_weight,
            )
            self.warmup = False

    def update_cache(self, indices: torch.Tensor):
        if self.use_cache:
            # pyre-fixme[16]
            tt_embeddings.update_cache_state(indices, self.hashtbl, self.cache_freq)

    def forward(
        self, indices: torch.Tensor, offsets: torch.Tensor, warmup: bool = True
    ) -> torch.Tensor:
        (indices, offsets) = indices.long(), offsets.long()

        # update hash table and lfu state
        self.update_cache(indices)

        # preprocess indices
        (
            indices,
            rowidx,
            tableidx,
            num_tt_indices,
            cache_locations,
            # pyre-fixme[16]
        ) = tt_embeddings.preprocess_indices_sync(
            indices,
            offsets,
            self.num_tables,
            self.warmup,
            # pyre-fixme[16]
            self.hashtbl,
            # pyre-fixme[16]
            self.cache_state,
        )
        num_cached = indices.numel() - num_tt_indices
        # pyre-fixme[16]
        output = TTLookupFunction.apply(
            # self.num_tables should be able to divide offsets.numel() - 1
            (offsets.numel() - 1) // self.num_tables,
            self.embedding_dim,
            self.tt_p_shapes,
            self.tt_q_shapes,
            self.tt_ranks,
            # pyre-fixme[16]
            self.L,
            num_tt_indices,
            num_cached,
            indices,
            rowidx,
            tableidx,
            self.optimizer,
            self.learning_rate,
            self.eps,
            self.sparse,
            cache_locations,
            self.cache_optimizer_state,
            self.cache_weight,
            list(self.optimizer_state),
            *(self.tt_cores),
        )

        return output

    def set_learning_rate(self, lr: float) -> None:
        """
        Sets the learning rate.
        """
        self.learning_rate = lr

    def get_params(self) -> List[torch.Tensor]:
        params = self.tt_cores
        if self.use_cache:
            params.append(self.cache_weight)
        return params


class TTEmbeddingBag(TableBatchedTTEmbeddingBag):
    """
    TTEmbedding lookup for exactly one table
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tt_ranks: List[int],
        tt_p_shapes: Optional[List[int]] = None,
        tt_q_shapes: Optional[List[int]] = None,
        optimizer: OptimType = OptimType.SGD,
        learning_rate: float = 0.1,
        eps: float = 1.0e-10,
        sparse: bool = True,
        use_cache: bool = True,
        cache_size: int = 0,
        hashtbl_size: int = 0,
        weight_dist: str = "approx-normal",
        enforce_embedding_dim: bool = False,
    ) -> None:
        super().__init__(
            1,  # num_tables = 1
            num_embeddings,
            embedding_dim,
            tt_ranks,
            tt_p_shapes,
            tt_q_shapes,
            optimizer,
            learning_rate,
            eps,
            sparse,
            use_cache,
            cache_size,
            hashtbl_size,
            weight_dist,
            enforce_embedding_dim,
        )

    def forward(
        self, indices: torch.Tensor, offsets: torch.Tensor, warmup: bool = True
    ) -> torch.Tensor:
        return super().forward(indices, offsets, warmup)[
            0
        ]  # there should be only one table
