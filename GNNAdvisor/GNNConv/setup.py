from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GNNAdvisor',
    ext_modules=[
        CUDAExtension(
        name='GNNAdvisor', 
        sources=[   
                    'GNNAdvisor.cpp', 
                    'cuCompactor.cu',
                    'cuda_error_check.cu',
                    'GNNAdvisor_kernel.cu'
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })