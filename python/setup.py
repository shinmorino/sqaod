from setuptools import setup, find_packages, Extension
import platform
import numpy


def new_cpu_ext(name, srcs) :
    ext_includes = [numpy.get_include(), '../libsqaod/include', '../libsqaod', '../libsqaod/eigen']
    ext = Extension(name, srcs,
                    include_dirs=ext_includes,
                    extra_compile_args = ['-std=c++11', '-Wno-format-security'],
                    extra_link_args = ['-L../libsqaod/.libs', '-lsqaod'])
    return ext

def new_cuda_ext(name, srcs) :
    ext_includes = [numpy.get_include(), '/usr/local/cuda-9.1/include', '../libsqaod/include', '../libsqaod', '../libsqaod/eigen', '../libsqaod/cub']
    ext = Extension(name, srcs,
                    include_dirs=ext_includes,
                    extra_compile_args = ['-std=c++11', '-Wno-format-security'],
                    extra_link_args = ['-L../libsqaod/.libs', '-lsqaod_cuda','-lsqaod'])
    return ext
    
ext_modules = []
if platform.system() != 'Windows' :
    ext_modules.append(new_cpu_ext('sqaod.cpu.cpu_dg_bf_searcher', ['sqaod/cpu/src/cpu_dg_bf_searcher.cpp']))
    ext_modules.append(new_cpu_ext('sqaod.cpu.cpu_dg_annealer', ['sqaod/cpu/src/cpu_dg_annealer.cpp']))
    ext_modules.append(new_cpu_ext('sqaod.cpu.cpu_bg_bf_searcher', ['sqaod/cpu/src/cpu_bg_bf_searcher.cpp']))
    ext_modules.append(new_cpu_ext('sqaod.cpu.cpu_bg_annealer', ['sqaod/cpu/src/cpu_bg_annealer.cpp']))
    ext_modules.append(new_cpu_ext('sqaod.cpu.cpu_formulas', ['sqaod/cpu/src/cpu_formulas.cpp']))

    ext_modules.append(new_cuda_ext('sqaod.cuda.cuda_device', ['sqaod/cuda/src/cuda_device.cpp']))
    ext_modules.append(new_cuda_ext('sqaod.cuda.cuda_dg_bf_searcher', ['sqaod/cuda/src/cuda_dg_bf_searcher.cpp']))
    ext_modules.append(new_cuda_ext('sqaod.cuda.cuda_dg_annealer', ['sqaod/cuda/src/cuda_dg_annealer.cpp']))
    ext_modules.append(new_cuda_ext('sqaod.cuda.cuda_bg_bf_searcher', ['sqaod/cuda/src/cuda_bg_bf_searcher.cpp']))
    ext_modules.append(new_cuda_ext('sqaod.cuda.cuda_bg_annealer', ['sqaod/cuda/src/cuda_bg_annealer.cpp']))
    ext_modules.append(new_cuda_ext('sqaod.cuda.cuda_formulas', ['sqaod/cuda/src/cuda_formulas.cpp']))

setup(
    name='sqaod',
    version='0.0.dev0',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.11'],
    author='Shinya Morino',
    author_email="shin.morino_at_gmail.com",
    description='A collection of solvers for Quantum annealer.',
    license='BSD 3-Clause License',
    keywords='quantum annealing solver',
    ext_modules=ext_modules,
)
