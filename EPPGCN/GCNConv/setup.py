from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='GNNAdvisor',
    ext_modules=[
        CUDAExtension(
        name='GNNAdvisor', 
        sources=[   
                    'EPPGCN.cpp', 
                    'cuCompactor.cu',
                    'cuda_error_check.cu',
                    'EPPGCN_kernel.cu'
                ],
        extra_compile_args={'cxx': ['-O3', '-fopenmp'],
                            'nvcc': []}
        ),
        # CppExtension(
        #     name='process',
        #     sources=[
        #         'process.cpp'
        #     ],
        #     extra_compile_args=['-O3', '-fopenmp']
        # )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })