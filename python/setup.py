from setuptools import setup, find_packages, Extension
import numpy


def new_ext(name, srcs) :
    ext_includes = [numpy.get_include(), '../libsqaod/include', '../libsqaod', '../libsqaod/eigen']
    ext = Extension(name, srcs,
                    include_dirs=ext_includes,
                    extra_compile_args = ['-std=c++11'],
                    extra_link_args = ['-L../libsqaod/.libs', '-lsqaod'])
    return ext
    
ext_modules = []
ext_modules.append(new_ext('sqaod.cpu.cpu_dg_bf_solver', ['sqaod/cpu/src/cpu_dg_bf_solver.cpp']))
ext_modules.append(new_ext('sqaod.cpu.cpu_dg_annealer', ['sqaod/cpu/src/cpu_dg_annealer.cpp']))
ext_modules.append(new_ext('sqaod.cpu.cpu_bg_bf_solver', ['sqaod/cpu/src/cpu_dg_bf_solver.cpp']))
ext_modules.append(new_ext('sqaod.cpu.cpu_bg_annealer', ['sqaod/cpu/src/cpu_bg_annealer.cpp']))
ext_modules.append(new_ext('sqaod.cpu.cpu_formulas', ['sqaod/cpu/src/cpu_formulas.cpp']))

setup(
    name='sqaod',
    version='0.0dev0',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.11'],
    author='Shinya Morino',
    author_email="shin.morino_at_gmail.com",
    description='A collection of solvers for Quantum annealer.',
    license='BSD 3-Clause License',
    keywords='quantum annealing solver',
    ext_modules=ext_modules,
)
