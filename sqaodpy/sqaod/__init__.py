from __future__ import print_function

__version__ = '0.3.1'

from .common.envcheck import *
from .common.cuda_probe import is_cuda_available

def run_env_check() :
    print('CUDA : {}'.format('Available' if is_cuda_available() else 'Not available'))

    checker = EnvChecker(__version__)
    checker.check()
    checker.show()

try :
    # imports
    from .common.preference import algorithm
    from .common.preference import minimize
    from .common.preference import maximize

    from .common import *
    from . import py
    from . import cpu

    if is_cuda_available() :
        from . import cuda
    
except:
    checker = EnvChecker(__version__)
    if not checker.check() :
        print('\nWrong installation of sqaod backend libraries.')
        print('Please visit https://github.com/shinmorino/sqaod/wiki/installation to check installation instruction.\n')
        checker.show()
