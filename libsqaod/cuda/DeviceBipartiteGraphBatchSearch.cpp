#include "DeviceBipartiteGraphBatchSearch.h"


// template<class real>
// void BGFuncs<real>::batchSearch(real *E, PackedBitsPairArray *xPairs,
//                                 const EigenDeviceMatrix &b0, const EigenDeviceMatrix &b1, const EigenDeviceMatrix &W,
//                                 PackedBits xBegin0, PackedBits xEnd0,
//                                 PackedBits xBegin1, PackedBits xEnd1) {
//     int nBatch0 = int(xEnd0 - xBegin0);
//     int nBatch1 = int(xEnd1 - xBegin1);

//     real Emin = *E;
//     int N0 = W.cols();
//     int N1 = W.rows();
//     EigenDeviceMatrix eBitsSeq0(nBatch0, N0);
//     EigenDeviceMatrix eBitsSeq1(nBatch1, N1);

//     createBitsSequence(eBitsSeq0.data(), N0, xBegin0, xEnd0);
//     createBitsSequence(eBitsSeq1.data(), N1, xBegin1, xEnd1);
    
//     EigenDeviceMatrix eEBatch = eBitsSeq1 * (W * eBitsSeq0.transpose());
//     eEBatch.rowwise() += (b0 * eBitsSeq0.transpose()).row(0);
//     eEBatch.colwise() += (b1 * eBitsSeq1.transpose()).transpose().col(0);
    
//     /* FIXME: Parallelize */
//     for (int idx1 = 0; idx1 < nBatch1; ++idx1) {
//         for (int idx0 = 0; idx0 < nBatch0; ++idx0) {
//             real Etmp = eEBatch(idx1, idx0);
//             if (Etmp > Emin) {
//                 continue;
//             }
//             else if (Etmp == Emin) {
//                 xPairs->push_back(PackedBitsPairArray::value_type(xBegin0 + idx0, xBegin1 + idx1));
//             }
//             else {
//                 Emin = Etmp;
//                 xPairs->clear();
//                 xPairs->push_back(PackedBitsPairArray::value_type(xBegin0 + idx0, xBegin1 + idx1));
//             }
//         }
//     }
//     *E = Emin;
// }
    

