#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension
import os
import pybind11
os.path.dirname(os.path.abspath(__file__))

setup(
    name="meshing_surfel_rasterization",
    packages=['meshing_surfel_rasterization',
              'TSDF_forGS' # 新增的包
              ],
    version='0.0.7',
    ext_modules=[
        CUDAExtension(
            name="meshing_surfel_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                                'nvcc': [
                                    '-I' + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                                        ]
                                }
            ),
        CppExtension(
            name='TSDF_forGS._C',
            sources=[
                "TSDF_forGS/TSDF.cpp",
                "TSDF_forGS/binding.cpp",
                "TSDF_forGS/cameras.cpp"
                     ],
            include_dirs=[
                pybind11.get_include(),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/") # 添加第三方库glm
                ],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
        )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
