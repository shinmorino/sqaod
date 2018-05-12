#pragma once

#include <thread>
#include <atomic>

namespace sqaod_internal {

class ParallelWorkDistributor {
public:
    
    ParallelWorkDistributor() {
        threads_ = NULL;
    }

    ~ParallelWorkDistributor() { }
    
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
                            ParallelWorkDistributor::threadEntry(this, idx + 1); });
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
    void run(F &f) {
        functor_ = std::move(f);
        if (nThreads_ != 0) {
            completionCounter_ = 0;
            nThreadsToRun_ = nThreads_;
            std::atomic_thread_fence(std::memory_order_release);
            
            /* run the 0-th worker in main thread. */
            functor_(0);

            while (true) {
                if (completionCounter_ == nThreads_)
                    break;
            }
        }
        else {
            functor_(0);
        }
    }
    
private:

    static
    void threadEntry(ParallelWorkDistributor *_this, int threadIdx) {
        _this->mainloop(threadIdx);
    }
    
    void mainloop(int threadIdx) {

        while (true) {
            while (run_) {
                int cur = nThreadsToRun_;
                if (0 < cur) {
                    if (nThreadsToRun_.compare_exchange_weak(cur, cur - 1))
                        break;
                }
            }
            if (!run_)
                break;
            
            functor_(threadIdx);
            
            std::atomic_thread_fence(std::memory_order_release); /* memory barrier
                                                                  * FIXME: is it required ? */
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
