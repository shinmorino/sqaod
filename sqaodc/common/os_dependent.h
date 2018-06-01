#pragma once

#include <cstddef>

namespace sqaod {

int getNumActiveCores();


void *aligned_alloc(int alignment, size_t size);

void aligned_free(void *pv);


}
