#include "UniformOp.h"
#include "EigenBridge.h"

using namespace sqaod;

double sqaod::sum(const double *values, SizeType cols, SizeType rows, SizeType stride) {
    if (rows == 1) {
        EigenMappedColumnVectorType<double> mapped(const_cast<double*>(values), cols);
        return mapped.sum();
    }
    else {
        EigenMappedMatrixType<double> mapped(const_cast<double*>(values),
                                             rows, cols, Eigen::OuterStride<>(stride));
        return mapped.sum();
    }
}

float sqaod::sum(const float *values, SizeType cols, SizeType rows, SizeType stride) {
    if (rows == 1) {
        EigenMappedColumnVectorType<float> mapped(const_cast<float*>(values), cols);
        return mapped.sum();
    }
    else {
        EigenMappedMatrixType<float> mapped(const_cast<float*>(values),
                                            rows, cols, Eigen::OuterStride<>(stride));
        return mapped.sum();
    }
}


