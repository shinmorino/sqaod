#include "MinimalTestSuite.h"
#include <iostream>

int MinimalTestSuite::okCount_ = 0;
int MinimalTestSuite::failCount_ = 0;

void MinimalTestSuite::success() {
    std::cerr << "." << std::flush;
    ++okCount_;
}

void MinimalTestSuite::fail(const char *filename, unsigned long lineno) {
    std::cerr << filename << ":" << lineno << ": Test failed(" << testName_ << ")" << std::endl;
    ++failCount_;
}

int MinimalTestSuite::summarize() {
    std::cerr << std::endl
              << std::endl
              << "FAILED: " << failCount_ << " / ALL: " << okCount_ + failCount_ << std::endl
              << std::endl;

    if (failCount_ == 0)
        std::cerr << "PASSED ALL TESTS." << std::endl;
    std::cerr << std::endl;

    if (failCount_ != 0)
        return 1;
    return 0;
}


bool compare(const DeviceImage<float> &gpu, const HostImage<float> &cpu, std::ostream &ostm) {

    if (gpu.width_ != cpu.width_) {
        ostm << "width does not match" << std::endl;
        return false;
    }
    if (gpu.height_ != cpu.height_) {
        ostm << "height does not match" << std::endl;
        return false;
    }

    HostImage<float> gpuRes;
    copy(&gpuRes, gpu);
    for (int iy = 0; iy < gpuRes.height_; ++iy) {
        for (int ix = 0; ix < gpuRes.width_; ++ix) {
            float vCPU = *cpu.getPtr(ix, iy);
            float vGPU = *gpuRes.getPtr(ix, iy);
            float diff = fabs((vGPU - vCPU) / vCPU);
            if (2.e-5f < diff) {
                ostm << "Error: (ix, iy)=(" << ix << "," << iy << ")";
                return false;
            }
        }
    }
    return true;
}


bool compare(const char *caption, const DeviceImage<float> &gpu, const HostImage<float> &cpu, float tolerance, std::ostream &ostm) {
    HostImage<float> gpuCopied;
    copy(&gpuCopied, gpu);

	char line[1024];
	bool errFound = false;
	float maxDiff = 0.f, maxNormDiff = 0.f;
	for (int iy = 0; iy < cpu.height_; ++iy) {
		for (int ix = 0; ix < cpu.width_; ++ix) {
			float vCPU = cpu.get(ix, iy);
			float vGPU = gpuCopied.get(ix, iy);
			float diff = fabs(vGPU - vCPU);
			float normDiff = (vGPU - vCPU) / vCPU;
			if ((tolerance < diff) && (tolerance < normDiff)) {
				errFound = true;
				if (tolerance < diff) {
					sprintf(line, "ERR : (%d, %d) GPU: %9.3g CPU: %9.3g Diff(absolute) %g\n", ix, iy, vGPU, vCPU, diff);
					ostm << line;
				}
				if (tolerance < normDiff) {
					sprintf(line, "ERR : (%d, %d) GPU: %9.3g CPU: %9.3g Diff(normalized) %g\n", ix, iy, vGPU, vCPU, normDiff);
					ostm << line;
				}
			}
			maxDiff = std::max(maxDiff, diff);
			maxNormDiff = std::max(maxNormDiff, normDiff);
		}
	}

	if (errFound) {
		sprintf(line, "ERROR %s, maxErr=%g.\n", caption, maxDiff);
		ostm << line;
		return false;
	}
	sprintf(line, "%s: maxErr=%g.\n", caption, maxDiff);
	ostm << line;
	return true;
}