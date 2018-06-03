#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>


namespace sqaod_internal {

class ParallelWorkDistributor_cpp {
public:
    
    ParallelWorkDistributor_cpp() {
        threads_ = NULL;
    }

    ~ParallelWorkDistributor_cpp() { }
    
    void initialize(int nWorkers) {
        nThreads_ = nWorkers - 1;
        mainThreadId_ = std::this_thread::get_id();
        if (0 < nThreads_) {
            threads_ = (std::thread*)malloc(sizeof(std::thread) * nThreads_);
            completionCounter_ = 0;
        }
    }
    
    void finalize() {
        if (nThreads_ != 0)
            free(threads_);
        threads_ = NULL;
    }
    
    template<class F>
    void run(F &f, int nWorkers = -1) {
        functor_ = std::move(f);
        if (nWorkers == -1)
            nWorkers = nThreads_ + 1;
        if (1 < nWorkers) {
            {
                for (int idx = 0; idx < nThreads_; ++idx) {
                    new (&threads_[idx]) std::thread([this, idx]{
                                ParallelWorkDistributor_cpp::threadEntry(this, idx + 1); });
                }
                for (int idx = 0; idx < nThreads_; ++idx)
                    threads_[idx].detach();
            }
            
            /* run the 0-th worker in main thread. */
            functor_(0);

             {
                 std::unique_lock<std::mutex> lock(mutex_);
                 compCond_.wait(lock, [this]{ return completionCounter_ == nThreads_; } );
             }
             
             for (int idx = 0; idx < nThreads_; ++idx)
                 threads_[idx].~thread();

             completionCounter_ = 0;
        }
        else {
            functor_(0);
        }
    }
    

private:

    static
    void threadEntry(ParallelWorkDistributor_cpp *_this, int threadIdx) {
        _this->runWorker(threadIdx);
    }
    
    void runWorker(int threadIdx) {

        functor_(threadIdx);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ++completionCounter_;
            if (completionCounter_ == nThreads_)
                compCond_.notify_one();
        }
        
    }

    std::thread *threads_;
    int nThreads_;
    std::thread::id mainThreadId_;
    std::function<void(int)> functor_;
    int nThreadsToRun_;
    int completionCounter_;
    std::condition_variable compCond_;
    std::mutex mutex_;
};

}
