import numpy as np
from . import preference as pref
import inspect

def inherit(dst_cls, src_cls) :
    src_dict = src_cls.__dict__
    dst_dict = dst_cls.__dict__
    
    dst_keys =  dst_dict.keys()
    for name, attr in src_dict.items() :
        if inspect.isfunction(attr) and ((name in dst_keys) and hasattr(attr, '__doc__')) :
            attr.__doc__ = src_dict[name].__doc__
            
def copy(dst, src) :
    dst.__doc__ = src.__doc__
