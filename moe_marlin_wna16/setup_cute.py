import os
from setuptools import setup
from torch.utils import cpp_extension

root_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(root_dir)

setup(
    name='moe_marlin_cute',
    version='0.1.0',
    description='MOE Marlin CuTe kernel.',
    ext_modules=[cpp_extension.CUDAExtension(
        'moe_marlin_cute_cuda',
        [os.path.join(root_dir, 'moe_cute_ops.cu')],
        include_dirs=[
            root_dir,
            os.path.join(parent_dir, '3rdparty/cutlass/include'),
        ],
        extra_compile_args={
            'nvcc': ['--expt-relaxed-constexpr', '-std=c++17'],
        },
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
