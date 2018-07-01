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
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
    'License :: OSI Approved :: BSD License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis'
]

classifiers = classifiers + pyver
url = 'https://github.com/shinmorino/sqaod/'

with open('doc/README.rst') as file:
    long_description = file.read()


setup(
    name=name,
    version='0.3.0',
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
    package_data=package_data,
    include_package_data=True,
    distclass=BinaryDistribution
)
