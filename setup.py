"""Build the C++ graph generator extension module."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "generator",
        ["search_ft/generator/generator.cpp"],
        include_dirs=["search_ft/generator"],
        cxx_std=11,
        extra_compile_args=["-Ofast", "-DNDEBUG", "-fno-stack-protector"],
    ),
]

setup(
    name="generator",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
