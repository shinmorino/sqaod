#ifndef COMMON_H__
#define COMMON_H__

#include <common/Matrix.h>


namespace sqaod {
    
    
enum OptimizeMethod {
    optMinimize,
    optMaximize
};

    
template<class real>
void createBitsSequence(real *bits, int nBits, int bBegin, int bEnd);
    
void unpackBits(Bits *unpacked, const PackedBits packed, int N);
    

template<class real>
bool isSymmetric(const MatrixType<real> &W);

}

#endif
