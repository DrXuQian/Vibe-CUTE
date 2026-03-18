import os
import glob
from setuptools import setup
from torch.utils import cpp_extension

root_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(root_dir)
vllm_csrc = os.path.join(parent_dir, '3rdparty/vllm/csrc')

# All generated kernel .cu files + ops.cu
cuda_sources = glob.glob(os.path.join(root_dir, 'sm80_kernel_*.cu'))
cuda_sources.append(os.path.join(root_dir, 'ops.cu'))

setup(
    name='moe_marlin',
    version='0.1.0',
    description='MOE Marlin WNA16 kernel from vLLM.',
    ext_modules=[cpp_extension.CUDAExtension(
        'moe_marlin_cuda',
        cuda_sources,
        include_dirs=[
            vllm_csrc,
            root_dir,
            os.path.join(parent_dir, '3rdparty/cutlass/include'),
        ],
        extra_compile_args={
            'nvcc': ['--expt-relaxed-constexpr', '-std=c++17',
                     '-DTORCH_EXTENSION_NAME=moe_marlin_cuda'],
        },
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
