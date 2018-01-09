#include "cudafuncs.h"
#include "DeviceMath.h"

using sqaod::SizeType;
using namespace sqaod_cuda;


namespace {

struct Assign {
    template<class real>
    __device__ real operator()(real &d_value, const real &v) {
        return d_value = v;
    }
};
        
template<class real>
struct AddAssignOp {
    AddAssignOp(real _mulFactor) : mulFactor(_mulFactor) { }
    __device__ real operator()(real &d_value, const real &v) {
        return d_value = d_value * mulFactor + v;
    }
    real mulFactor;
};

template<class real>
AddAssignOp<real> AddAssign(real mulFactor) {
    return AddAssignOp<real>(mulFactor);
}


}


template<class real>  static __global__
void scaleKernel(real *d_y, real alpha, const real *d_x, SizeType size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        d_y[gid] += alpha * d_x[gid];
}

template<class real>
void DeviceMathType<real>::scale(real *d_y, real alpha, const real *d_x, SizeType size) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    addKernel<<<gridDim, blockDim>>>(d_y, alpha, d_x, size);
    DEBUG_SYNC;
}

template<class real, template<class> class assignOp>
static __global__
void scaleBroadcastKernel(real *d_y, real alpha, const real *d_c, SizeType size,
                          assignOp<real> assign) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size)
        assign(d_y[gid], alpha * *d_c);
}

template<class real>
void DeviceMathType<real>::scaleBroadcast(real *d_y, real alpha, const real *d_c, SizeType size,
                                          real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x));
    if (addAssignFactor == 0.)
        scaleBroadcastKernel<<<gridDim, blockDim>>>(d_y, alpha, d_c, size, Assign());
    else
        scaleBroadcastKernel<<<gridDim, blockDim>>>(d_y, alpha, d_c, size,
                                                    AddAssign(addAssignFactor));
    DEBUG_SYNC;
}

template<class real, template<class> class assignOp>  static __global__
void scaleBroadcastVectorKernel(real *d_A, real alpha, const real *d_x, SizeType size,
                                assignOp<real> assign) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx < size) {
        SizeType pos = gidx + size * gidy;
        assign(d_A[pos], alpha * d_x[gidx]);
    }
}

template<class real>
void DeviceMathType<real>::scaleBroadcastVector(real *d_A, real alpha, const real *d_x, SizeType size,
                                                SizeType nBatch, real addAssignFactor) {
    dim3 blockDim(128);
    dim3 gridDim(divru(size, blockDim.x), divru(nBatch, blockDim.y));
    if (addAssignFactor == 0.)
        scaleBroadcastVector<<<gridDim, blockDim>>>(d_A, alpha, d_x, size, Assign());
    else
        scaleBroadcastVector<<<gridDim, blockDim>>>(d_A, alpha, d_x, size,
                                                    AddAsign(addAssignFactor));
    DEBUG_SYNC;
}

