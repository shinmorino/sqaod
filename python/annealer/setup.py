from distutils.core import setup, Extension
import numpy

module = Extension('native',
                   sources = ['native.c'],
                   include_dirs=[numpy.get_include()])

setup (name = 'native',
       version = '0.1',
       description = 'native extension for annealer',
       ext_modules = [module])

setup(
    name='annealer',
    #version='0.0.1',
    description='',
    #long_description=readme,
    #author='',
    #author_email='',
    #install_requires=['numpy'],
    requires=['numpy'],
    #url='https://github.com/kennethreitz/samplemod',
    #license=license,
    #packages=packages(exclude=('tests', 'docs'))
    #packages=find_packages(exclude=('tests', 'docs'))
)
