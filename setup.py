import shlex
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def pkg_config(option: str) -> list[str]:
    output = subprocess.check_output(["pkg-config", option, "libpcre2-8"], text=True)
    return shlex.split(output)


include_dirs = [flag[2:] for flag in pkg_config("--cflags-only-I")]
library_dirs = [flag[2:] for flag in pkg_config("--libs-only-L")]
libraries = [flag[2:] for flag in pkg_config("--libs-only-l")]

extension = Pybind11Extension(
    "chatbot._tokenizer_cpp",
    ["cpp/tokenizer.cpp", "cpp/bindings.cpp"],
    include_dirs=["cpp", *include_dirs],
    library_dirs=library_dirs,
    libraries=libraries,
    cxx_std=20,
    extra_compile_args=["-O3"],
)

setup(
    ext_modules=[extension],
    cmdclass={"build_ext": build_ext},
)
