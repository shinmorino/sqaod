// https://qiita.com/g0117736/items/a82604d0725e8a7ab821
#pragma once

#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <limits.h>

namespace sqaod_internal {

inline
int futex(int *uaddr, int op, int val) {
    return syscall(SYS_futex, uaddr, op, val, NULL, NULL, 0);
}

class ParallelWorkDistributor_linux {
public:
    
    ParallelWorkDistributor_linux() {
        threads_ = NULL;
    }

    ~ParallelWorkDistributor_linux() {
        finalize();
    }
    
    void initialize(int nDefaultWorkers) {
        pthread_attr_init(&thread_attr_);
        pthread_attr_setinheritsched(&thread_attr_, PTHREAD_EXPLICIT_SCHED);
        mainThreadId_ = pthread_self();
        run_ = true;

        nDefaultWorkers_ = nDefaultWorkers;
        nMaxWorkers_ = std::max(4, nDefaultWorkers);
        nThreads_ = nMaxWorkers_ - 1;
        if (0 < nThreads_) {
            nThreadsToRun_ = 0;
            completionCounter_ = 0;
            __atomic_thread_fence(__ATOMIC_RELEASE);

            threads_ = new pthread_t[nThreads_];
            for (int idx = 0; idx < nThreads_; ++idx) {
                pthread_create(&threads_[idx], &thread_attr_,
                               ParallelWorkDistributor_linux::threadEntry, this);
            }
        }
    }
    
    void finalize() {
        run_ = false;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        futex(&completionCounter_, FUTEX_WAKE_PRIVATE, INT_MAX);
        
        delete [] threads_;
        threads_ = NULL;
    }
    
    template<class F>
    void run(F &f, int nWorkers = -1) {
        functor_ = f;
        nWorkers_ = nWorkers;
        if (nWorkers_ == -1)
            nWorkers_ = nDefaultWorkers_;
        if (1 < nWorkers_) {
            __atomic_store_n(&nThreadsToRun_, nWorkers_ - 1, __ATOMIC_RELEASE);
            futex(&nThreadsToRun_, FUTEX_WAKE_PRIVATE, nWorkers_ - 1); /* wake n threads */
            
            /* run the 0-th worker in main thread. */
            functor_(0);
            
            int counter = __atomic_add_fetch(&completionCounter_, 1, __ATOMIC_RELAXED);
            int fastPathCount = 0;
            while (counter != nWorkers_) {                
                counter = __atomic_load_n(&completionCounter_, __ATOMIC_RELAXED);
                if (fastPathCount < 500) {
                    ++fastPathCount;
                    continue;
                }
                _mm_pause();
            }            
            completionCounter_ = 0;
        }
        else {
            functor_(0);
        }
    }

private:

    static
    void *threadEntry(void *pvarg) {
        ParallelWorkDistributor_linux *_this = (ParallelWorkDistributor_linux*)pvarg;
        _this->runWorker();
        return NULL;
    }
    
    void runWorker() {
        while (run_) {
            int threadIdx;
            while (run_) {
                threadIdx = __atomic_load_n(&nThreadsToRun_, __ATOMIC_RELAXED);
                if (0 < threadIdx) {
                    bool success =
                            __atomic_compare_exchange_n(&nThreadsToRun_, &threadIdx, threadIdx - 1,
                                                        true, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
                    if (success)
                        break;
                }
                else {
                    futex(&nThreadsToRun_, FUTEX_WAIT_PRIVATE, 0);
                }
            }
            
            if (!run_)
                break;
            
            functor_(threadIdx);
            
             __atomic_add_fetch(&completionCounter_, 1, __ATOMIC_RELAXED);
        }
    }

    std::function<void(int)> functor_;

    pthread_t *threads_;
    pthread_attr_t thread_attr_;
    int nThreads_;
    pthread_t mainThreadId_;
    bool run_;
    
    int nThreadsToRun_;
    int completionCounter_;

    int nDefaultWorkers_;
    int nMaxWorkers_;
    int nWorkers_;
};

}
