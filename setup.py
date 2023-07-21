from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

with open("README.md") as f:
    readme = f.read()

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "funi",
        ["src/pyfuni.cpp"],
        include_dirs=["src"],
        extra_compile_args=["-O3"],
        cxx_std=17,
    )
]

setup(
    name="funi",
    version=__version__,
    description="Find unique float arrays",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jaewook Lee",
    author_email="jaewooklee042@gmail.com",
    url="https://github.com/tataratat/funi",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    package_data={"src": ["*.hpp"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "numpy",
    ],
    zip_safe=False,
    license="MIT",
)
