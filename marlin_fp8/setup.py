import os
import glob
from setuptools import setup
from torch.utils import cpp_extension

root_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(root_dir)
gen_dir = os.path.join(root_dir, 'generated')

cuda_sources = glob.glob(os.path.join(gen_dir, 'sm80_kernel_*.cu'))
cuda_sources.append(os.path.join(root_dir, 'marlin.cu'))

setup(
    name='marlin_fp8',
    version='0.1.0',
    description='Marlin FP8 (E4M3) GEMM kernel from vLLM.',
    ext_modules=[cpp_extension.CUDAExtension(
        'marlin_fp8_cuda',
        cuda_sources,
        include_dirs=[root_dir, gen_dir],
        extra_compile_args={
            'nvcc': ['--expt-relaxed-constexpr', '-std=c++17',
                     '-DTORCH_EXTENSION_NAME=marlin_fp8_cuda'],
        },
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
