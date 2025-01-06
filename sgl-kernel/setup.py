from pathlib import Path

import os
import sys
import shutil
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()

def is_cuda() -> bool:
    """Return whether it is CUDA on the NVIDIA CUDA platform."""
    return torch.cuda.is_available() and torch.version.cuda


def is_hip() -> bool:
    """Return whether it is HIP on the AMD ROCm platform."""
    return torch.cuda.is_available() and torch.version.hip


def get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


def update_wheel_platform_tag():
    wheel_dir = Path("dist")
    old_wheel = next(wheel_dir.glob("*.whl"))
    new_wheel = wheel_dir / old_wheel.name.replace(
        "linux_x86_64", "manylinux2014_x86_64"
    )
    old_wheel.rename(new_wheel)

if is_cuda():
    cutlass = root / "3rdparty" / "cutlass"
    include_dirs = [
        cutlass.resolve() / "include",
        cutlass.resolve() / "tools" / "util" / "include",
    ]
    nvcc_flags = [
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    cxx_flags = ["-O3"]
    libraries = ["c10", "torch", "torch_python"]
    extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib"]
    ext_modules = [
        CUDAExtension(
            name="sgl_kernel.ops._kernels",
            sources=[
                "src/sgl-kernel/csrc/trt_reduce_internal.cu",
                "src/sgl-kernel/csrc/trt_reduce_kernel.cu",
                "src/sgl-kernel/csrc/moe_align_kernel.cu",
                "src/sgl-kernel/csrc/sgl_kernel_ops.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "nvcc": nvcc_flags,
                "cxx": cxx_flags,
            },
            libraries=libraries,
            extra_link_args=extra_link_args,
        ),
    ]
else:
    def validate_and_update_archs(archs):
        # List of allowed architectures
        allowed_archs = ["native", "gfx90a",
                         "gfx940", "gfx941", "gfx942", "gfx1100"]

        # Validate if each element in archs is in allowed_archs
        assert all(
            arch in allowed_archs for arch in archs
        ), f"One of GPU archs of {archs} is invalid or not supported"

    def rename_cpp_to_cu(els, dst, recurisve=False):
        def do_rename_and_mv(name, src, dst, ret):
            newName = name
            if name.endswith(".cpp") or name.endswith(".cu"):
                newName = name.replace(".cpp", ".cu")
                ret.append(f'{dst}/{newName}')
            shutil.copy(f'{src}/{name}', f'{dst}/{newName}')
        ret = []
        for el in els:
            if not os.path.exists(el):
                continue
            if os.path.isdir(el):
                for entry in os.listdir(el):
                    if os.path.isdir(f'{el}/{entry}'):
                        if recurisve:
                            ret += rename_cpp_to_cu([f'{el}/{entry}'],
                                                    dst, recurisve)
                        continue
                    do_rename_and_mv(entry, el, dst, ret)
            else:
                do_rename_and_mv(os.path.basename(el),
                                 os.path.dirname(el), dst, ret)
        return ret

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ck_dir = os.environ.get("CK_DIR", f"{root}/3rdparty/composable_kernel")
    bd_dir = f"{this_dir}/build"

    if not os.path.exists(bd_dir):
        os.makedirs(bd_dir)

    shutil.copytree(ck_dir, f'{bd_dir}/ck', dirs_exist_ok=True)

    ck_dir = f'{bd_dir}/ck'
    
    archs = os.getenv("GPU_ARCHS", "gfx942").split(";")
    validate_and_update_archs(archs)
    
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

    cc_flag += [
        "-mllvm", "-enable-post-misched=0",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        "-mllvm", "--amdgpu-kernarg-preload-count=16",
        "-mllvm", "-amdgpu-coerce-illegal-types=1",
        "-Wno-unused-result",
        "-Wno-switch-bool",
        "-Wno-vla-cxx-extension",
        "-Wno-undefined-func-template",
    ]

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc":
            [
                "-O3", "-std=c++17",
                "-fPIC",
                "-DUSE_PROF_API=1",
                "-DENABLE_FP8",
                "-DUSE_ROCM",
                "-D__HIP_PLATFORM_HCC__=1",
                "-D__HIP_PLATFORM_AMD__=1",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF_OPERATORS__",
        ]
            + cc_flag,
    }

    include_dirs = [
            f"{this_dir}/build",
            f"{ck_dir}/include",
            f"{ck_dir}/library/include",
            f"{ck_dir}/example/ck_tile/15_fused_moe",
    ]

    renamed_ck_srcs = rename_cpp_to_cu(
    [  # f'for other kernels'
        f"{ck_dir}/example/ck_tile/15_fused_moe/instances",
    ], bd_dir)

    build_srcs = ["src/sgl-kernel/csrc/moe_align_kernel.cu",
            "src/sgl-kernel/csrc/moe_fused_experts.cu",
            "src/sgl-kernel/csrc/sgl_kernel_ops.cu"]

    ext_modules = [
        CUDAExtension(
            name="sgl_kernel.ops._kernels",
            sources=build_srcs+renamed_ck_srcs,
            extra_compile_args=extra_compile_args,
            libraries=["hiprtc", "amdhip64", "c10", "torch", "torch_python"],
            include_dirs=include_dirs,
            extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        ),
    ]

setup(
    name="sgl-kernel",
    version=get_version(),
    packages=["sgl_kernel"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)

update_wheel_platform_tag()
