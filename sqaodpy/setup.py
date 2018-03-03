from setuptools import setup, find_packages, Extension, dist
import sys

#https://stackoverflow.com/questions/35112511/pip-setup-py-bdist-wheel-no-longer-builds-forced-non-pure-wheels
class BinaryDistribution(dist.Distribution):
    def has_ext_modules(self):
        return True
    
if sys.platform == 'linux2' :
    package_data = {'sqaod.cpu' : ['*.so', 'sqaod/cpu/*.so' ], 'sqaod.cuda' : ['*.so', 'sqaod/cuda/*.so' ] }
else :    
    package_data = {'sqaod.cpu' : ['*.pyd', 'sqaod/cpu/*.pyd' ], 'sqaod.cuda' : ['*.pyd', 'sqaod/cuda/*.pyd' ] }
    
setup(
    name='sqaod',
    version='0.1.dev0',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.11'],
    author='Shinya Morino',
    author_email="shin.morino_at_gmail.com",
    description='A collection of solvers for simulated quantum annealer.',
    license='Apache2',
    keywords='quantum annealing solver',
    #ext_modules=ext_modules,
    package_data=package_data,
    include_package_data=True,
    distclass=BinaryDistribution
)
