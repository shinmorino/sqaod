from __future__ import print_function
from cuda_probe import is_cuda_available, cuda_failure_reason 
from ctypes.util import find_library

msg = """
The sqaod backend native library have not been found.
Please visit https://github.com/shinmorino/wiki/Installation to confirm installation instruction.
"""[1:-1]

def envcheck() :
    res = find_library('sqaodc')
    if res is not None :
        return True
    print(msg)
    return False

def show_env() :
    sqaodc_installed = find_library('sqaodc')
    sqaodc_cuda_installed = find_library('sqaodc_cuda')
    cuda_available = is_cuda_available()

    print('sqaodc CPU  library : {}'.format('OK' if sqaodc_installed is not None else 'not found'))
    print('sqaodc CUDA library : {}'.format('OK' if sqaodc_cuda_installed is not None else 'not found'))
    print('CUDA                : {}'.format('Available' if sqaodc_cuda_installed else 'not available'))
          

if __name__ == '__main__' :
    envcheck()
    show_env()

