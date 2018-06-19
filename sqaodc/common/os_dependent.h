#pragma once

#include <cstddef>

namespace sqaod {

int getDefaultNumThreads();


void *aligned_alloc(int alignment, size_t size);

void aligned_free(void *pv);


}
