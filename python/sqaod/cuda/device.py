import cuda_device

class Device :
    def __init__(self, devno) :
        # make ext as a member, and it should be initialized, 
        # since exception may occur in the nextline.
        self._ext = None
        self._ext = cuda_device.device_new(devno)

    def __del__(self) :
        if not self._ext is None :
            self.finalize();

    def finalize(self) :
        cuda_device.device_finalize(self._ext);
        cuda_device.device_delete(self._ext)
        self._ext = None
        
def create_device(devno = 0) :
    return Device(devno)


# Global/Static

def unload(active_device) :
    if not active_device is None :
    	active_device.finalize()
	active_device = None

if __name__ != "__main__" :
    active_device = create_device()
    import atexit
    atexit.register(unload, active_device)

