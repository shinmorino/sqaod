import sysconfig
import numpy
import subprocess

_DEBUG=False

# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", '-fPIC']
if _DEBUG:
    extra_compile_args += ["-O0", "-ggdb"]
else:
    extra_compile_args += ["-DNDEBUG", "-O3"]

include_dirs=sysconfig.get_config_var('INCLDIRSTOMAKE').split()
include_dirs.append(numpy.get_include())
include_dirs = [ '-I' + dir for dir in include_dirs]

sources = ['native.c', 'mt19937ar.c']

command = ['gcc', '-shared'] + extra_compile_args + include_dirs + sources + ['-o', '../native.so']
print(command)
print(' '.join(command))
subprocess.check_call(command)

