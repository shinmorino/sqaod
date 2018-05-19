#include <sqaodc.h>
#include <sqaodc/cpu/Dot_SIMD.h>
#include <sqaodc/common/EigenBridge.h>
#include <chrono>
#include <iostream>
#include <random>

namespace sq = sqaod;

template<class T>
void showDuration(const T &duration) {
    std::cout << "elapsed time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " msec."
              << std::endl;
}

int main() {
    typedef sq::EigenRowVectorType<float> Vector;
    int size = 32; //1 << 10;
    
    std::mt19937 gen(16423);

    Vector v0(size);
    Vector v1(size);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int idx = 0; idx < size; ++idx) {
        v0(idx) = dist(gen);
        v1(idx) = dist(gen);
    }
    
    auto start = std::chrono::system_clock::now();
    float dot0 = sqaod_cpu::dot_sse2(v0.data(), v1.data(), size);
    auto end = std::chrono::system_clock::now();
    float dot1 = v0.dot(v1);
    printf("%g %g %g\n", dot0, dot1, dot0 - dot1);
    
    showDuration(end - start);
}
