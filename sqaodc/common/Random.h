/* -*- c++ -*- */
#pragma once

namespace sqaod {

class Random {
public:
    Random();

    void seed();

    void seed(unsigned long long s);

    void initByArray(unsigned long initKey[], int keyLength);

    unsigned long randInt32();

    unsigned long randInt(int N);

    double randomf64();

    float randomf32();

    template<class real>
    real random();
    
    
private:
    enum {
        N = 624,
        M = 397
    };

    unsigned long mt[N]; /* the array for the state vector  */
    int mti; /* mti==N+1 means mt[N] is not initialized */
};



template<> inline
float Random::random<float>() {
    return randomf32();
}
template<> inline
double Random::random<double>() {
    return randomf64();
}

extern Random random;

}

