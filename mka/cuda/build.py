from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_nvcc = ["-O3", "--use_fast_math", "-lineinfo"]
_cxx = ["-O3"]

setup(
    name="mka_cuda",
    ext_modules=[
        CUDAExtension(
            name="fastmka_cuda",
            sources=["fastmka_attn.cpp", "fastmka_attn.cu"],
            extra_compile_args={"cxx": _cxx, "nvcc": _nvcc},
        ),
        CUDAExtension(
            name="fused_route_mka_cuda",
            sources=["fused_route_mka.cpp", "fused_route_mka.cu"],
            extra_compile_args={"cxx": _cxx, "nvcc": _nvcc},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
