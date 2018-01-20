#ifndef TESTS_H__
#define TESTS_H__

#include "MinimalTestSuite.h"


class Tests : public MinimalTestSuite {
public:
    Tests(void);
    ~Tests(void);

    virtual void setUp();

    virtual void tearDown();
    
    virtual void run(std::ostream &ostm);

};

#endif
