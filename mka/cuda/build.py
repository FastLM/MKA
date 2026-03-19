from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fastmka_cuda",
    ext_modules=[
        CUDAExtension(
            name="fastmka_cuda",
            sources=["fastmka_attn.cpp", "fastmka_attn.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
