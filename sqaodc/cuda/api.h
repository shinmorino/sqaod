#pragma once

#include <sqaodc/common/Solver.h>
#include <sqaodc/common/Formulas.h>


extern "C"
void sqaodc_cuda_version(int *version, int *cuda_version);


namespace sqaod {

namespace cuda {


class Device : public NullBase {
public:
    Device() { }

    virtual ~Device() { }

    virtual void initialize(int devNo = 0) = 0;

    virtual void finalize() = 0;

private:
    Device(const Device &);
};


struct DeviceAssigner {

    virtual ~DeviceAssigner() WAR_VC_NOTHROW { }
    
    virtual void assignDevice(Device &device) = 0;
    
};


template<class real>
struct DenseGraphAnnealer : DeviceAssigner, sqaod::DenseGraphAnnealer<real> {

    virtual ~DenseGraphAnnealer() WAR_VC_NOTHROW { }

};

template<class real>
struct BipartiteGraphAnnealer : DeviceAssigner, sqaod::BipartiteGraphAnnealer<real> {

    virtual ~BipartiteGraphAnnealer() WAR_VC_NOTHROW { }

};

template<class real>
struct DenseGraphBFSearcher : DeviceAssigner, sqaod::DenseGraphBFSearcher<real> {

    virtual ~DenseGraphBFSearcher() WAR_VC_NOTHROW { }

};

template<class real>
struct BipartiteGraphBFSearcher :
            DeviceAssigner, sqaod::BipartiteGraphBFSearcher<real> {

    virtual ~BipartiteGraphBFSearcher()  { }

};


template<class real>
struct DenseGraphFormulas : DeviceAssigner, sqaod::DenseGraphFormulas<real> {

    virtual ~DenseGraphFormulas()  { }

};


template<class real>
struct BipartiteGraphFormulas : DeviceAssigner, sqaod::BipartiteGraphFormulas<real> {

    virtual ~BipartiteGraphFormulas()  { }

};

}

}
