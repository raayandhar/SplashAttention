# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='splash_attention',
    ext_modules=[
        CUDAExtension('splash_attention', [
            'splash.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
