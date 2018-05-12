#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>



namespace sqaod_internal {

class ParallelWorkDistributor_Lock {
public:
    
    ParallelWorkDistributor_Lock() {
        threads_ = NULL;
    }

    ~ParallelWorkDistributor_Lock() { }
    
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
                            ParallelWorkDistributor_Lock::threadEntry(this, idx + 1); });
            }
        }
    }
    
    void joinThreads() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            run_ = false;
            runCond_.notify_all();
        }
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
            {
                std::unique_lock<std::mutex> lock(mutex_);
                nThreadsToRun_ = nWorkers - 1;
                runCond_.notify_all();
            }
            
            /* run the 0-th worker in main thread. */
            functor_(0);

             {
                 std::unique_lock<std::mutex> lock(mutex_);
                compCond_.wait(lock, [this]{ return completionCounter_ == nThreads_; } );
            }
            completionCounter_ = 0;
        }
        else {
            functor_(0);
        }
    }
    
private:

    static
    void threadEntry(ParallelWorkDistributor_Lock *_this, int threadIdx) {
        _this->mainloop(threadIdx);
    }
    
    void mainloop(int threadIdx) {

        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                runCond_.wait(lock, [this]{ return (!run_) || (nThreadsToRun_ != 0); });
                --nThreadsToRun_;
            }
            if (!run_)
                break;
            
            functor_(threadIdx);
            
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ++completionCounter_;
                compCond_.notify_one();
            }
        }
        
    }

    std::thread *threads_;
    int nThreads_;
    bool run_;
    std::function<void(int)> functor_;
    int nThreadsToRun_;
    int completionCounter_;
    std::condition_variable runCond_;
    std::condition_variable compCond_;
    std::mutex mutex_;
};

}
