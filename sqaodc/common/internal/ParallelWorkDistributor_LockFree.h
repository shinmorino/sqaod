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
        mainThreadId_ = pthread_self();
        if (0 < nThreads_) {
            threads_ = (std::thread*)malloc(sizeof(std::thread) * nThreads_);
            memset((void*)triggered_, 0, sizeof(int64_t) * (nThreads_ + 1));
            completionCounter_ = 0;
            run_ = true;
            std::atomic_thread_fence(std::memory_order_release); /* memory barrier */
        
            for (int idx = 0; idx < nThreads_; ++idx) {
                new (&threads_[idx]) std::thread([this, idx]{
                            ParallelWorkDistributor_LockFree::threadEntry(this, idx + 1); });
            }
        }
    }
    
    void finalize() {
        run_ = false;
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

            while (true) {
                if (completionCounter_ == nThreads_)
                    break;
            }
            completionCounter_ = 0;
        }
        else {
            functor_(0);
        }
    }

    void barrier() {
        // std::atomic_thread_fence(std::memory_order_release);
                
        if (mainThreadId_ == pthread_self()) {
            while (true) {
                if (completionCounter_ == nThreads_)
                    break;
            }
            completionCounter_ = 0;
        }
        else {
            __sync_fetch_and_add(&completionCounter_, 1);
            std::atomic_thread_fence(std::memory_order_release);
            /* wait for main thread to clear completionCounter. */
            while (true) {
                if (completionCounter_ == 0)
                    break;
            }
        }
    }
    
private:

    static
    void threadEntry(ParallelWorkDistributor_LockFree *_this, int threadIdx) {
        _this->mainloop(threadIdx);
    }
    
    void mainloop(int threadIdx) {

        while (true) {
            while (run_) {
                if (triggered_[threadIdx] != 0)
                    break;
                _mm_pause();
            }
            triggered_[threadIdx] = 0;
            if (!run_)
                break;
            
            functor_(threadIdx);
            __sync_fetch_and_add(&completionCounter_, 1);
        }
        
    }

    std::thread *threads_;
    int nThreads_;
    pthread_t mainThreadId_;
    volatile bool run_;
    std::function<void(int)> functor_;
    volatile int64_t triggered_[32];
    volatile int64_t completionCounter_;
};

}
