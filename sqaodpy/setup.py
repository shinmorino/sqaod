from setuptools import setup, find_packages, Extension, dist
import sys

#https://stackoverflow.com/questions/35112511/pip-setup-py-bdist-wheel-no-longer-builds-forced-non-pure-wheels
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(self):
        return True

    
if 'linux' in sys.platform :
    package_data = {'sqaod.cpu' : ['*.so', 'sqaod/cpu/*.so' ], 'sqaod.cuda' : ['*.so', 'sqaod/cuda/*.so' ] }
else :    
    package_data = {'sqaod.cpu' : ['*.pyd', 'sqaod/cpu/*.pyd' ], 'sqaod.cuda' : ['*.pyd', 'sqaod/cuda/*.pyd' ] }
    

name = 'sqaod'

pyver= [
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
]

email = 'shin.morino@gmail.com'    
author='Shinya Morino'


classifiers=[
    'Operating System :: POSIX :: Linux',
    'Natural Language :: English',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Information Analysis'
]

classifiers = classifiers + pyver
url = 'https://github.com/shinmorino/sqaod/'

setup(
    name=name,
    version='0.1.1a1',
    url=url,
    author=author,
    author_email=email,
    maintainer=author,
    maintainer_email=email,
    description='A collection of solvers for simulated quantum annealing.',
    long_description='Sqaod is a collection of sovlers for simulated quantum annealing.  Solvers are acelerated by using OpenMP on multi-core CPUs and by using CUDA on NVIDIA GPUs.' +
    'Please visit sqaod website for details, ' + url + '.',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.11'],
    keywords='Simulated quantum annealing, Quantum computing, Monte Carlo, OpenMP, GPU, CUDA',
    classifiers=classifiers,
    package_data=package_data,
    include_package_data=True,
    distclass=BinaryDistribution
)
