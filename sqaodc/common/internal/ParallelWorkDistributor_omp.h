#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <omp.h>

namespace sqaod_internal {

class ParallelWorkDistributor_omp {
public:
    
    ParallelWorkDistributor_omp() {
    }

    ~ParallelWorkDistributor_omp() { }
    
    void initialize(int nWorkers) {
    }
    
    void finalize() {
    }
    
    template<class F>
    void run(F &f, int nWorkers = 2) {
        functor_ = std::move(f);
        if (1 < nWorkers) {
#pragma omp parallel
            {
                functor_(omp_get_thread_num());
            }
        }
        else {
            functor_(0);
        }
    }
    

private:
    std::function<void(int)> functor_;
};

}
