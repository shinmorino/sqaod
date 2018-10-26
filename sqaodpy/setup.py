from setuptools import setup, find_packages, Extension, dist
from sysconfig import get_config_vars
import numpy as np
import sys
import os
from ctypes.util import find_library

name = 'sqaod'
version = '1.0.0'

npinclude = np.get_include()

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)


def extension(pkg, name) :
    ext = Extension('sqaod/{}/{}'.format(pkg, name),
                    include_dirs = [npinclude],
                    libraries = ['sqaodc'],
                    library_dirs = ['/usr/lib', 'sqaod/{}'.format(pkg)],
                    sources = ['sqaod/{}/src/{}.cpp'.format(pkg, name)],
                    extra_compile_args = ['-std=c++11', '-Wno-format-security'])
    return ext

def cuda_extension(pkg, name) :
    ext = Extension('sqaod/{}/{}'.format(pkg, name),
                    include_dirs = [npinclude],
                    libraries = ['sqaodc_cuda', 'sqaodc'],
                    library_dirs = ['/usr/lib', 'sqaod/{}'.format(pkg)],
                    sources = ['sqaod/{}/src/{}.cpp'.format(pkg, name)],
                    extra_compile_args = ['-std=c++11', '-Wno-format-security'])
    return ext

ext_modules = []
ext_modules.append(extension('cpu', 'cpu_formulas'))
ext_modules.append(extension('cpu', 'cpu_dg_bf_searcher'))
ext_modules.append(extension('cpu', 'cpu_dg_annealer'))
ext_modules.append(extension('cpu', 'cpu_bg_bf_searcher'))
ext_modules.append(extension('cpu', 'cpu_bg_annealer'))
if find_library('sqaodc_cuda') is not None :
    ext_modules.append(cuda_extension('cuda', 'cuda_device'))
    ext_modules.append(cuda_extension('cuda', 'cuda_formulas'))
    ext_modules.append(cuda_extension('cuda', 'cuda_dg_bf_searcher'))
    ext_modules.append(cuda_extension('cuda', 'cuda_dg_annealer'))
    ext_modules.append(cuda_extension('cuda', 'cuda_bg_bf_searcher'))
    ext_modules.append(cuda_extension('cuda', 'cuda_bg_annealer'))

pyver= [
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

email = 'shin.morino@gmail.com'    
author='Shinya Morino'


classifiers=[
    'Operating System :: POSIX :: Linux',
    'Natural Language :: English',
    'License :: OSI Approved :: Apache Software License',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
    'License :: OSI Approved :: BSD License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis'
]

classifiers = classifiers + pyver
url = 'https://github.com/shinmorino/sqaod/'

with open('README.rst') as file:
    long_description = file.read()

setup(
    name=name,
    version=version,
    url=url,
    author=author,
    author_email=email,
    maintainer=author,
    maintainer_email=email,
    description='A collection of solvers for simulated quantum annealing.',
    long_description=long_description,
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.11'],
    keywords='Simulated quantum annealing, Quantum annealing, Quantum computing, Monte Carlo, OpenMP, GPU, CUDA',
    classifiers=classifiers,
    ext_modules=ext_modules
)
