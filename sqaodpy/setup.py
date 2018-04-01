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
    
isPy2 = sys.version_info[0] == 2

name = 'sqaod-py2' if isPy2 else 'sqaod-py3'
if isPy2 :
    pyver= [
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ]
else :
    pyver= [
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

setup(
    name=name,
    version='0.1.0',
    url='https://github.com/shinmorino/sqaod/',
    author=author,
    author_email=email,
    maintainer=author,
    maintainer_email=email,
    description='A collection of solvers for simulated quantum annealer.',
    long_description='',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.11'],
    keywords='simulated quantum annealing, quantum computing, Monte-Carlo simulation, CPU, GPU, CUDA',
    classifiers=classifiers,
    package_data=package_data,
    include_package_data=True,
    distclass=BinaryDistribution
)

