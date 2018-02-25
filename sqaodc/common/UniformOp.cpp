#include "UniformOp.h"
#include "EigenBridge.h"

using namespace sqaod;

double sqaod::sum(const double *values, SizeType size) {
    EigenMappedColumnVectorType<double> mapped(const_cast<double*>(values), size);
    return mapped.sum();
}

float sqaod::sum(const float *values, SizeType size) {
    EigenMappedColumnVectorType<float> mapped(const_cast<float*>(values), size);
    return mapped.sum();
}


