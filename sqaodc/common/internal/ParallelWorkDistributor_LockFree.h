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
    
    void initialize(int nWorkers) {
        nThreads_ = nWorkers - 1;
        mainThreadId_ = std::this_thread::get_id();
        if (0 < nThreads_) {
            threads_ = (std::thread*)malloc(sizeof(std::thread) * nThreads_);
            memset((void*)triggered_, 0, sizeof(int64_t) * (nThreads_ + 1));
            run_ = true;
            std::atomic_store_explicit(&completionCounter_,  0, std::memory_order_release);
        
            for (int idx = 0; idx < nThreads_; ++idx) {
                new (&threads_[idx]) std::thread([this, idx]{
                            ParallelWorkDistributor_LockFree::threadEntry(this, idx + 1); });
            }
        }
    }
    
    void finalize() {
        run_ = false;
        std::atomic_thread_fence(std::memory_order_release); /* memory barrier */
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
            for (int idx = 0; idx < nThreads_ + 1; ++idx)
                triggered_[idx] = 1;
            std::atomic_thread_fence(std::memory_order_release);
            
            /* run the 0-th worker in main thread. */
            functor_(0);

            int fastPath = 0;
            while (true) {
                if (completionCounter_ == nThreads_)
                    break;
                ++fastPath;
                if (fastPath < 300)
                    continue;
                _mm_pause();
            }
            std::atomic_store_explicit(&completionCounter_, 0, std::memory_order_release);
        }
        else {
            functor_(0);
        }
    }
    
private:

    static
    void threadEntry(ParallelWorkDistributor_LockFree *_this, int threadIdx) {
        _this->mainloop(threadIdx);
    }
    
    void mainloop(int threadIdx) {

        while (run_) {
            while (run_) {
                if (triggered_[threadIdx] != 0)
                    break;
                _mm_pause();
            }
            triggered_[threadIdx] = 0;
            if (!run_)
                break;
            
            functor_(threadIdx);
            std::atomic_fetch_add_explicit(&completionCounter_, 1, std::memory_order_relaxed);
        }
        
    }

    std::thread *threads_;
    int nThreads_;
    std::thread::id mainThreadId_;
    bool run_;
    std::function<void(int)> functor_;
    volatile int64_t triggered_[32];
    std::atomic_int completionCounter_;
};

}
