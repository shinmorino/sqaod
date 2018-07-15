from __future__ import print_function
import sys

def is_cuda_available() :
    try :
        # loading driver
        import sqaod.cuda
    except :
        return False
    else :
        return True

def cuda_failure_reason() :
    try :
        # try loading cuda module.
        import sqaod.cuda
    except RuntimeError as e :
        print(e)
    except ImportError as e :
        print(e)
    except :
        print("Unexpected error:", sys.exc_info())
    else :
        return true, "OK"
        
if __name__ == '__main__' :
    cuda_available = is_cuda_available()
    print('cuda available : ' + str(cuda_available))
    if cuda_available is False :
        cuda_failure_reason()
