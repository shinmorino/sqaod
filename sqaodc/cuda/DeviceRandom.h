#pragma once

#include <sqaodc/cuda/DeviceRandomMTGP32.h>
#include <sqaodc/cuda/DeviceRandomMT19937.h>

namespace sqaod_cuda {

// typedef DeviceRandomMTGP32 DeviceRandom;
typedef DeviceRandomMT19937 DeviceRandom;

}

