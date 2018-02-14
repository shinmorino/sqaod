import cuda_device

class Device :
    def __init__(self, devno) :
        self.ext = cuda_device.device_new(devno)

    def __del__(self) :
        if not self.ext is None :
            self.device_finalize(ext);

    def finalize(self) :
        cuda_device.device_finalize(self.ext);
        cuda_device.device_delete(self.ext)
        self.ext = None

        
def create_device(devno = 0) :
    Device.device = Device(devno)
    return Device.device

if __name__ == "__main__" :
    dev = create_device()
    dev.finalize()
