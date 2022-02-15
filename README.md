# FBTT-Embedding
`FBTT-Embedding` library provides functionality to compress sparse embedding tables commonly used in machine learning models such as recommendation and natural language processing. The library can be used as a direct replacement to [PyTorch’s EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) functionality. It provides the forward and backward propagation functionality same as PyTorch’s `EmbeddingBag` with only difference of compression.
In addition, our implementation includes a software cache to store a portion of the entries in the embedding tables (or “bag”s) in decompressed format for faster lookup and process removing the need for decompressing and compressing the entries every-time it is accessed during the program execution of training or inference.

Read more at ["TT-Rec: Tensor Train Compression for Deep Learning Recommendation Models"](https://proceedings.mlsys.org/paper/2021/file/979d472a84804b9f647bc185a877a8b5-Paper.pdf), in the Proceedings of Conference on Machine Learning and Systems, [MLSys 2021](https://mlsys.org/).

## Installing FBTT-Embedding

* The implementation was compiled and tested with PyTorch 1.6 and above, with CUDA 10.1.
* To install the library, run setup.py install
    In order to get the best performance, specify `-gencode for nvcc` in the `setup.py` (line 24). The corresponding compiler settings for each architecture can be found in CUDA Toolkit Documentation, for example
    Volta https://docs.nvidia.com/cuda/volta-compatibility-guide/index.html
    Pascal https://docs.nvidia.com/cuda/pascal-compatibility-guide/index.html
* Install dependence cub: https://nvlabs.github.io/cub/
* A sample run of the code, run `tt_embeddings_benchmark.py`. A sample output is shown below

``` bash
INFO:root:sparse: True, optimizer: sgd
INFO:root:p_shapes: [200, 220, 250], q_shapes: [4, 4, 4], ranks: [32, 32]
INFO:root:B: 512, E: 11000000, D: 64, nnz: 10240
INFO:root:TTEmbeddingBag FWD-BWD time/nnz: 0.416 usecs, GFLOPS: 2657.631, BW: 18.456
```

## How FBTT-Embedding works

### Parameters

* `num_embeddings(int)` — size of the dictionary of embeddings
* `embedding_dim(int)` — the length of each embedding vector
* `tt_ranks(List[int])` — the ranks of TT cores
* `tt_p_shapes(Optional[List[int]])` — The factorization of `num_embeddings`, where the product of all elements is not smaller than `num_embeddings`.
* `tt_q_shapes(Optional[List[int]])` — The factorization of `embedding_dim`, where the product of all elements is equal to `embedding_dim`.
* `sparse (bool)` — if True, the weight update is fused with gradient calculation, and gradients are not returned by backward propagation. Otherwise the gradient w.r.t. TT cores or cache would be returned to external optimizer.
* `optimizer (OptimType)` — The type of optimizer when using fused kernel.
* `learning_rate (float)` — Learning rate of the optimizer
* `eps (float)` — term added to the denominator to improve numerical stability, for Adagrad only.
* `use_cache (bool)` — if True, a software cache will be used to store the most-frequently-accessed embedding vectors.
* `cache_size (int)` — The maximum number of embedding vectors to be stored in the cache.
* `hashtbl_size (int`) — The maximum number of entries in the hash table for frequency count.
* `weight_dist (str)` — `“uniform”`, `“naive-uniform”`, `“approx-uniform”`, `"normal"`, `“approx-normal”`. When using `“uniform”` or `“normal”`, the weights of TT cores will be i.i.d from the specified distribution. When using `“approx-uniform”` or `“approx-normal”`, the TT cores are initialized in a way that the entries of full embedding table follow the specified distribution.

### Initialization
The initialization of TT-Emb is similar to Pytorch EmbeddingBag

``` python
tt_emb = TTEmbeddingBag(
        num_embeddings=1000000,
        embedding_dim=64,
        tt_p_shapes=[120, 90, 110],
        tt_q_shapes=[4, 4, 4],
        tt_ranks=[12, 14],
        sparse=False,
        use_cache=False,
        weight_dist="uniform"
    )
```

This method will generate TT cores representing an embedding table of size 1000000 by 64 (`num_embeddings x embedding_dim`), where each TT core is of size `ranks[i] x tt_p_shapes[i] x tt_q_shapes[i] x ranks[i+1] and ranks = [1]+tt_rank+[1]`. In this case, the shape of the 3 TT-cores are `1 x 120 x 4 x 12, 12 x 90 x 4 x 16`, and `14 x 110 x 4 x 1`.
When `tt_p_shapes` and `tt_q_shapes are` specified, the product of `tt_p_shapes[]` needs be no smaller than `num_embeddings`; the product of `tt_q_shapes` must be equal to `embedding_dim`. When passing these 2 parameters as None, TTEmbeddingBag will factor `num_embeddings` and `embedding_dim` automatically.

``` python
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding_sum = TTEmbeddingBag(10, 3, None, None, tt_ranks=[2,2], sparse=False, use_cache=False)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([1,2,4,5,4,3,2,9])
>>> offsets = torch.LongTensor([0,4])
>>> embedding_sum(input, offsets)
tensor([[-0.8861, -5.4350, -0.0523],
[ 1.1306, -2.5798, -1.0044]])
```

### Fused Kernel
TT-Emb supports fused gradient computation and weight updates for better efficiency, where the weights of embedding tables are updated along with backward propagation. If the network is trained with an external optimizer, the gradients will no longer be returned to the optimizer. To enable the fused kernel, specify sparse=True, and pass the corresponding optimizer type and parameters to TTEmbeddingBag. For example,

``` python
tt_emb = TTEmbeddingBag(
        num_embeddings=1000000,
        embedding_dim=64,
        tt_p_shapes=[120, 90, 110],
        tt_q_shapes=[4, 4, 4],
        tt_ranks=[12, 14],
        sparse=True,
        optimizer=OptimType.SGD,
        learning_rate=0.05,
        eps=1.0e-10 #for ADAGRAD only
        use_cache=False,
    )
```

### Software Cache
Embedding lookup in TT-Rec requires explicitly computing the embedding vectors from TT cores via two consecutive matrix multiplications (GEMM). Similarly in backward propagation, the gradient of each tensor core is calculated through a chain of matrix multiplications in reversed order.
To reduce the computation during training, TT-Emb implements a software cache to store an uncompressed copy of the most-frequently queried embedding vectors. When such vectors are queried, the vectors can be loaded directly from cache without computation. The size of cache can be determined according to each hardware platform and dataset, so that cached embedding rows capture as many embedding lookups as possible while minimizing the memory requirement during training.
We implemented a 32-way set-associative Least-Frequently-Used (LFU) cache using open addressing hash table for frequency count. To enable cache for TT-Emb, specify `use_cache=True` and set the appropriate cache size as the maximum number of embedding vectors to store in the cache, and max hash table size.

``` python
tt_emb = TTEmbeddingBag(
        num_embeddings=1000000,
        embedding_dim=64,
        tt_p_shapes=[120, 90, 110],
        tt_q_shapes=[4, 4, 4],
        tt_ranks=[12, 14],
        sparse=True,
        optimizer=OptimType.SGD,
        learning_rate=0.05,
        eps=1.0e-10 #for ADAGRAD only
        use_cache=True,
        cache_size=1000,
        hashtbl_size=1000
    )
```

During forward propagation, the access frequency of embedding vectors will be updated. However, the cache will only be updated when `tt_emb.cache_populate()` is called. The cached rows are determined by the access frequency of the embedding vectors, and the value of each embedding vector would be initialized from TT cores.

### TableBatchedTTEmbeddingBag

Apart from `TTEmbeddingBag` demonstrated above, the library also includes `TableBatchedTTEmbeddingBag` which groups the lookup operation together for multiple TT embedding tables. This can be more efficient than creating multiple instances of `TTEmbeddingBag`s when each of the `TTEmbeddingBag` has little computation involved.

To use that, you simply need to initialize it as below:

``` python
tt_emb = TableBatchedTTEmbeddingBag(
        num_tables=100,
        num_embeddings=1000000,
        embedding_dim=64,
        tt_p_shapes=[120, 90, 110],
        tt_q_shapes=[4, 4, 4],
        tt_ranks=[12, 14],
        sparse=False,
        use_cache=False,
        weight_dist="uniform"
    )
```

The above creates 100 TT embedding tables with the same dimensions underlying. The only additional argument that needs to be passed is `num_tables`

Currently there are some limitations to `TableBatchedTTEmbeddingBag`.

* The mutiple tables in `TableBatchedTTEmbeddingBag` must share the same dimensions.
* No support for software cache yet.
* No support for `"approx-uniform"` weight init yet.
* No support for expanding to full weight yet.

## License
FBTT-Embedding is MIT licensed, as found in the LICENSE file.
