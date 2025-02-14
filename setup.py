from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Path to the PyTorch include directory
import torch
torch_include_path = torch.utils.cpp_extension.include_paths()

setup(
    name='sparse_attention',
    ext_modules=[
        CUDAExtension(
            name='sparse_attention',
            sources=['sparse_attention.cu'],
            include_dirs=torch_include_path,  # Explicitly add PyTorch's include paths
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_86,code=sm_86',  # For RTX 3060
                    '-gencode=arch=compute_86,code=compute_86',
                    '--use_fast_math',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
