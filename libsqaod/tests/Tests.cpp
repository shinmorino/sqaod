#include "Tests.h"
#include <cuda_runtime.h>

Tests::Tests(void) : MinimalTestSuite("Tests")
{
}


Tests::~Tests(void)
{
}


void Tests::setUp() {
}

void Tests::tearDown() {
}
    
void Tests::run(std::ostream &ostm) {
    TEST_ASSERT(true);
}
