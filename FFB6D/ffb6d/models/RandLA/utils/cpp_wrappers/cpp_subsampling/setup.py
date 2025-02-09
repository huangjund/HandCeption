from distutils.core import setup, Extension
import numpy.distutils.misc_util
import os

# Specify the desired GCC version (GCC 14)
gcc_version = 'gcc-14'  # or 'g++-14' for g++

# Ensure that the correct GCC version is used
os.environ["CC"] = gcc_version
os.environ["CXX"] = gcc_version

m_name = "grid_subsampling"

SOURCES = ["../cpp_utils/cloud/cloud.cpp",
           "grid_subsampling/grid_subsampling.cpp",
           "wrapper.cpp"]

# Add additional flags to ensure compatibility with the required libstdc++ version
module = Extension(m_name,
                   sources=SOURCES,
                   extra_compile_args=['-std=c++11',
                                       '-D_GLIBCXX_USE_CXX11_ABI=0',
                                       '-fPIC'],  # Add -fPIC for position-independent code
                   extra_link_args=['-L/usr/lib/x86_64-linux-gnu', '-lstdc++'],  # Link against the correct libstdc++
                   include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())

setup(ext_modules=[module], include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())
