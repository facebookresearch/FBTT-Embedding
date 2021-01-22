# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

tnn_utils = CUDAExtension(
                        name='tt_emb',
                        sources=['tt_embeddings.cpp',
                            'tt_embeddings_cuda.cu',
                        ],
                        extra_compile_args={
                                'cxx': [
                                    "-O3",
                                    "-g",
                                    "-DUSE_MKL",
                                    "-m64",
                                    "-mfma",
                                    "-masm=intel",
                                    ],
                                'nvcc':[
                                    "-O3",
                                    "-g",
                                    "--expt-relaxed-constexpr",
                                    "-D__CUDA_NO_HALF_OPERATORS__",
                                    "-I/usr/lib/cub-1.8.0",
                                    "-gencode=arch=compute_70,code=\"sm_70\"",]
                        }
)
setup(name='tt_embeddings',
    description='tt_embeddings',
    packages=find_packages(),
    ext_modules=[tnn_utils],
    cmdclass={'build_ext':BuildExtension},
)
