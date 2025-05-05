import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# Check if torch is available
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class CustomBuildExt(build_ext):
    def run(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to build this extension. Please install PyTorch first.")
        super().run()

if TORCH_AVAILABLE:
    print(f"Building with PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

    ext_modules = [
        CUDAExtension('alt_cuda_corr',
                      sources=['correlation.cpp', 'correlation_kernel.cu'],
                      extra_compile_args={
                          'cxx': ['-O3'],
                          'nvcc': ['-O3', '--expt-relaxed-constexpr']
                      },
                      define_macros=[('WITH_CUDA', None)]
                      ),
    ]
    cmdclass = {
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
else:
    ext_modules = []
    cmdclass = {}

setup(
    name='alt_cuda_corr',
    version='0.1',
    description='Alternate CUDA correlation implementation',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.10.0',
    ],
)