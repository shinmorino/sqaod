/* -*- c++ -*- */
#ifndef CPU_RANDOM_H__
#define CPU_RANDOM_H__

class Random {
public:
    Random() {
        mti = N + 1;
    }

    void seed(unsigned long s);

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


#endif
