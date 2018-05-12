#pragma once

#include <thread>
#include <atomic>
#include <xmmintrin.h>


namespace sqaod_internal {

class ParallelWorkDistributor_LockFree {
public:
    
    ParallelWorkDistributor_LockFree() {
        threads_ = NULL;
    }

    ~ParallelWorkDistributor_LockFree() { }
    
    void runThreads(int nWorkers) {
        nThreads_ = nWorkers - 1;
        if (0 < nThreads_) {
            threads_ = (std::thread*)malloc(sizeof(std::thread) * nThreads_);
            nThreadsToRun_ = 0;
            completionCounter_ = 0;
            run_ = true;
            std::atomic_thread_fence(std::memory_order_release); /* memory barrier */
        
            for (int idx = 0; idx < nThreads_; ++idx) {
                new (&threads_[idx]) std::thread([this, idx]{
                            ParallelWorkDistributor_LockFree::threadEntry(this); });
            }
        }
    }
    
    void joinThreads() {
        *((volatile bool*)&run_) = false;
        if (nThreads_ != 0) {
            for (int idx = 0; idx < nThreads_; ++idx) {
                threads_[idx].join();
                threads_[idx].~thread();
            }
            free(threads_);
        }
        threads_ = NULL;
    }
    
    template<class F>
    void run(F &f, int nWorkers = -1) {
        functor_ = std::move(f);
        if (nWorkers == -1)
            nWorkers = nThreads_ + 1;
        if (1 < nWorkers) {
            std::atomic_thread_fence(std::memory_order_release);
            nThreadsToRun_ = nWorkers - 1;
            
            /* run the 0-th worker in main thread. */
            functor_(0);

            while (true) {
                if (completionCounter_ == nThreads_)
                    break;
            }
            completionCounter_ = 0;
            std::atomic_thread_fence(std::memory_order_release); /* memory barrier */
        }
        else {
            functor_(0);
        }
    }
    
private:

    static
    void threadEntry(ParallelWorkDistributor_LockFree *_this) {
        _this->mainloop();
    }
    
    void mainloop() {

        while (true) {
            int threadIdx;
            while (run_) {
                threadIdx = nThreadsToRun_;
                if (0 < threadIdx) {
                    if (nThreadsToRun_.compare_exchange_strong(threadIdx, threadIdx - 1)) {
                        break;
                    }
                }
                _mm_pause();
            }
            if (!run_)
                break;
            
            functor_(threadIdx);
            ++completionCounter_;
        }
        
    }

    std::thread *threads_;
    int nThreads_;
    std::atomic_bool run_;
    std::function<void(int)> functor_;
    std::atomic_int nThreadsToRun_;
    std::atomic_int completionCounter_;
};

}
