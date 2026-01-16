from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the ``get_include()`` method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/O2', '/arch:AVX2'],
        'unix': ['-O3', '-march=native', '-std=c++17'],
        'mingw32': ['-O3', '-mavx2', '-std=c++17']
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        elif ct == 'mingw32':
             opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            
        for ext in self.extensions:
            ext.extra_compile_args = opts
        
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "fast_ops",
        ["fast_ops.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            str(get_pybind_include()),
            str(get_pybind_include()) + '/user'
        ],
        language='c++'
    ),
]

setup(
    name="fast_ops",
    version="1.0",
    author="Paradox AI",
    description="C++ Accelerated Operations for Paradox AI Framework",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    headers=["fast_ops.h"],
    zip_safe=False,
)
