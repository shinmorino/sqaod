#include "CUDADenseGraphBFSolverTest.h"
#include <cpu/CPUFormulas.h>
#include <cpu/CPUDenseGraphBFSearcher.h>
#include <cpu/CPUDenseGraphBatchSearch.h>
#include <cuda_runtime.h>
#include "utils.h"

namespace sqcpu = sqaod_cpu;
namespace sqcu = sqaod_cuda;

CUDADenseGraphBFSolverTest::CUDADenseGraphBFSolverTest(void) : MinimalTestSuite("TCUDADenseGraphBFSolverTest")
{
}


CUDADenseGraphBFSolverTest::~CUDADenseGraphBFSolverTest(void)
{
}


void CUDADenseGraphBFSolverTest::setUp() {
    device_.useManagedMemory(true);
    device_.initialize();
}

void CUDADenseGraphBFSolverTest::tearDown() {
    device_.finalize();
}
    
void CUDADenseGraphBFSolverTest::run(std::ostream &ostm) {
    tests<double>();
    tests<float>();
}

template<class real>
void CUDADenseGraphBFSolverTest::tests() {
    typedef sq::MatrixType<real> Matrix;
    typedef sqcu::DeviceMatrixType<real> DeviceMatrix;

    sqcu::DeviceStream *devStream = device_.defaultStream();
    sqcu::DeviceObjectAllocator *devAlloc = device_.objectAllocator();
    sqcu::DeviceCopy devCopy(device_);

    testcase("test bit set 63") {
        sqcu::DeviceDenseGraphBatchSearch<real> search;
        search.assignDevice(device_);

        const sq::SizeType N = 63;
        DeviceMatrix d_bitsSequence;
        devAlloc->allocate(&d_bitsSequence, 1, N);
        Matrix bitsSequence;

        bool ok = true;
        for (int idx = 0; idx < (int)N; ++idx) {
            sq::PackedBitSet bits = 1ull << idx;
            search.generateBitsSequence(&d_bitsSequence, bits, bits + 1);
            devCopy(&bitsSequence, d_bitsSequence);
            devStream->synchronize();
            for (int chkIdx = 0; chkIdx < (int)N; ++chkIdx) {
                if (idx == int(N - 1 - chkIdx))
                    ok &= (bitsSequence.data[chkIdx] == real(1.));
                else
                    ok &= (bitsSequence.data[chkIdx] == real(0.));
            }
        }
        TEST_ASSERT(ok);
        devAlloc->deallocate(d_bitsSequence);
    }
    testcase("test bit set 32") {
        sqcu::DeviceDenseGraphBatchSearch<real> search;
        search.assignDevice(device_);

        const sq::SizeType N = 32;
        DeviceMatrix d_bitsSequence;
        devAlloc->allocate(&d_bitsSequence, 1, N);
        Matrix bitsSequence;

        bool ok = true;
        for (int idx = 0; idx < (int)N; ++idx) {
            sq::PackedBitSet bits = 1ull << idx;
            search.generateBitsSequence(&d_bitsSequence, bits, bits + 1);
            devCopy(&bitsSequence, d_bitsSequence);
            devStream->synchronize();
            for (int chkIdx = 0; chkIdx < (int)N; ++chkIdx) {
                if (idx == int(N - 1 - chkIdx))
                    ok &= (bitsSequence.data[chkIdx] == real(1.));
                else
                    ok &= (bitsSequence.data[chkIdx] == real(0.));
            }
        }
        TEST_ASSERT(ok);
        devAlloc->deallocate(d_bitsSequence);
    }

    testcase("test generate sequence 63") {
        sqcu::DeviceDenseGraphBatchSearch<real> search;
        search.assignDevice(device_);

        const sq::SizeType N = 63;
        const sq::SizeType tileSize = 1 << 12;
        DeviceMatrix d_bitsSequence;
        devAlloc->allocate(&d_bitsSequence, tileSize, N);
        Matrix bitsSequence;

        const sq::PackedBitSet xBegin = 1ull << 33;
        const sq::PackedBitSet xEnd = xBegin + tileSize;
        search.generateBitsSequence(&d_bitsSequence, xBegin, xEnd);
        devCopy(&bitsSequence, d_bitsSequence);
        devStream->synchronize();

        bool ok = true;
        for (sq::PackedBitSet seq = 0; seq < tileSize; ++seq) {
            sq::PackedBitSet bits = xBegin + seq;
            real *valseq = &bitsSequence(seq, 0);
            for (int idx = 0; idx < (int)N; ++idx) {
                bool bitSet = (bits & (1ull << (N - 1 - idx))) != 0;
                real expected = bitSet ? real(1.) : real(0.);
                if (expected != valseq[idx])
                    ok = false;
            }
        }
        TEST_ASSERT(ok);
        devAlloc->deallocate(d_bitsSequence);
    }

    testcase("test generate sequence 24") {
        sqcu::DeviceDenseGraphBatchSearch<real> search;
        search.assignDevice(device_);

        const sq::SizeType N = 24;
        const sq::SizeType tileSize = 1 << 12;
        DeviceMatrix d_bitsSequence;
        devAlloc->allocate(&d_bitsSequence, tileSize, N);
        Matrix bitsSequence;

        const sq::PackedBitSet xBegin = 1ull << 33;
        const sq::PackedBitSet xEnd = xBegin + tileSize;
        search.generateBitsSequence(&d_bitsSequence, xBegin, xEnd);
        devCopy(&bitsSequence, d_bitsSequence);
        devStream->synchronize();

        bool ok = true;
        for (sq::PackedBitSet seq = 0; seq < tileSize; ++seq) {
            sq::PackedBitSet bits = xBegin + seq;
            real *valseq = &bitsSequence(seq, 0);
            for (int idx = 0; idx < (int)N; ++idx) {
                bool bitSet = (bits & (1ull << (N - 1 - idx))) != 0;
                real expected = bitSet ? real(1.) : real(0.);
                if (expected != valseq[idx])
                    ok = false;
            }
        }
        TEST_ASSERT(ok);
        devAlloc->deallocate(d_bitsSequence);
    }

    testcase("partition") {
        sqcu::DeviceDenseGraphBatchSearch<real> search;
        search.assignDevice(device_);

        sqaod::SizeType N = 2048;
        real vToPart = 0.;

        sqaod::PackedBitSet *d_bitsListPart;
        sqaod::SizeType *d_nPart;
        real *d_values;

        devAlloc->allocate(&d_bitsListPart, N);
        devAlloc->allocate(&d_values, N);
        devAlloc->allocate(&d_nPart, 1);
        devStream->synchronize();
        for (sqaod::IdxType idx = 0; idx < (sqaod::IdxType)N; ++idx) {
            d_values[idx] = (real)(idx % 100 - 50);
        }

        sqaod::SizeType nPart = 0;
        for (int idx = 0; idx < (int)N; ++idx) {
            if (d_values[idx] == vToPart)
                ++nPart;
        }

        search.select(d_bitsListPart, d_nPart, 0, vToPart, d_values, N);
        devStream->synchronize();
        TEST_ASSERT(nPart == *d_nPart);
    }

    testcase("DenseGraphBatchSearch") {
        sq::SizeType N = 20;
        int tileSize = 4096;
        Matrix W = testMatSymmetric<real>(N);
        sqcpu::CPUDenseGraphBatchSearch<real> batchSearch;
        batchSearch.setQUBO(W, tileSize);
        batchSearch.initSearch();
        batchSearch.searchRange(0, tileSize);
        real E = batchSearch.Emin_;
        sq::PackedBitSetArray xList = batchSearch.packedXList_;

        sqcu::DeviceDenseGraphBatchSearch<real> devBatchSearch;
        devBatchSearch.assignDevice(device_);
        devBatchSearch.setQUBO(W, tileSize);
        devBatchSearch.calculate_E(0, tileSize);
        devBatchSearch.synchronize();
        TEST_ASSERT(E == devBatchSearch.get_Emin());

        devBatchSearch.partition_xMins(false);
        devBatchSearch.synchronize();

        TEST_ASSERT(devBatchSearch.get_xMins() == xList);

        devBatchSearch.partition_xMins(true);
        devBatchSearch.synchronize(); // FIXME: add hook ??
        sq::PackedBitSetArray xList2;
        xList2.insert(xList.begin(), xList.end());
        xList2.insert(xList.begin(), xList.end());
        TEST_ASSERT(devBatchSearch.get_xMins() == xList2)
    }

    testcase("minimum run") {
        int N = 16;
        Matrix W = testMatSymmetric<real>(N);
        TEST_ASSERT(sqaod::isSymmetric(W));

        sqcpu::CPUDenseGraphBFSearcher<real> cpuSolver;
        cpuSolver.setQUBO(W);
        cpuSolver.search();
        const sq::VectorType<real> &cpuE = cpuSolver.get_E();
        const sq::BitSetArray &cpuX = cpuSolver.get_x();

        sqcu::CUDADenseGraphBFSearcher<real> cudaSolver;
        cudaSolver.assignDevice(device_);
        cudaSolver.setQUBO(W);
        cudaSolver.search();
        const sq::VectorType<real> &cudaE = cudaSolver.get_E();
        const sq::BitSetArray &cudaX = cudaSolver.get_x();
        TEST_ASSERT(cpuE == cudaE);
        TEST_ASSERT(cpuX == cudaX);

        sq::VectorType<real> cpuVecX = sq::cast<real>(cpuX[0]);
        sq::VectorType<real> cudaVecX = sq::cast<real>(cudaX[0]);

        //real valCpuE, valCudaE;
        //DGFuncs<real>::calculate_E(&valCpuE, W, cpuVecX);
        //DGFuncs<real>::calculate_E(&valCudaE, W, cudaVecX);
        //TEST_ASSERT(valCpuE == valCudaE);
        // std::cerr << cpuX[0];
        // std::cerr << cudaX[0];
    }
}
