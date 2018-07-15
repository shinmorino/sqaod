try :
    # imports
    from .common.cuda_probe import is_cuda_available
    from .common.preference import algorithm
    from .common.preference import minimize
    from .common.preference import maximize
    from .common.envcheck import *

    from .common import *
    from . import py
    from . import cpu

    if is_cuda_available() :
        from . import cuda
    
except:
    from .common.envcheck import *
    envcheck()
