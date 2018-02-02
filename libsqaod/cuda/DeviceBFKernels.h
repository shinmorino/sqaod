#pragma once

#include <common/defines.h>
#include <cuda/DeviceStream.h>

namespace sqaod_cuda {

class Device;

template<class real>
struct DeviceBFKernelsType {

    void select(sqaod::PackedBits *d_out, sqaod::SizeType *d_nOut, sqaod::PackedBits xBegin, 
                real val, const real *d_vals, sqaod::SizeType nIn);

    DeviceBFKernelsType(DeviceStream *devStream = NULL);

    void assignStream(DeviceStream *devStream);
    
private:
    cudaStream_t stream_;
    DeviceStream *devStream_;
};

}

