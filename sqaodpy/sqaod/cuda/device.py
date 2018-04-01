from . import cuda_device
import sys

class Device :

    _cext = cuda_device

    def __init__(self, devno) :
        self._cobj = cuda_device.new()
        cuda_device.initialize(self._cobj, devno)

    def __del__(self) :
        if not self._cobj is None :
            self.finalize();
        self._cobj = None

    def finalize(self) :
        cuda_device.finalize(self._cobj);
        cuda_device.delete(self._cobj)
        self._cobj = None
        
def create_device(devno = 0) :
    return Device(devno)


# Global/Static

this_module = sys.modules[__name__]

def unload() :
    if not this_module.active_device is None :
        this_module.active_device = None

if __name__ != "__main__" :
    this_module.active_device = create_device()
    import atexit
    atexit.register(unload)
