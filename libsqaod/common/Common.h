#ifndef COMMON_H__
#define COMMON_H__

#include <common/Matrix.h>
#include <common/Array.h>

namespace sqaod {
    
    
enum OptimizeMethod {
    optMinimize,
    optMaximize
};

enum AnnealerState {
    annNone = 0,
    annRandSeedGiven = 1,
    annNTrottersGiven = 2,
    annQSet = 4,
};

    
template<class real>
void createBitsSequence(real *bits, int nBits, PackedBits bBegin, PackedBits bEnd);
    
void unpackBits(Bits *unpacked, const PackedBits packed, int N);
    

template<class real>
bool isSymmetric(const MatrixType<real> &W);

}

#endif
