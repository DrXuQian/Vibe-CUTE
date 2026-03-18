import os
from setuptools import setup
from torch.utils import cpp_extension

root_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(root_dir)

setup(
    name='marlin',
    version='0.1.1',
    description='Marlin FP16xINT4 CUDA matmul kernel (CuTe version).',
    install_requires=['numpy', 'torch'],
    packages=['.'],
    package_dir={'.': '.'},
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_cuda',
        ['marlin_cuda.cpp', 'marlin_cuda_kernel.cu'],
        include_dirs=[os.path.join(parent_dir, '3rdparty/cutlass/include')],
        extra_compile_args={'nvcc': ['--expt-relaxed-constexpr']},
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
