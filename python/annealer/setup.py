from distutils.core import setup, Extension
import sysconfig
import numpy

_DEBUG=True

# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", "-Wextra"]
if _DEBUG:
    extra_compile_args += ["-O0", "-ggdb"]
else:
    extra_compile_args += ["-DNDEBUG", "-O3"]

module = Extension('native',
                   sources = ['native.c', 'src/mt19937ar.c'],
                   extra_compile_args = extra_compile_args, 
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
