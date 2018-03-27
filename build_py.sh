#!/bin/bash

cd sqaodpy/sqaod/cpu/src
python incpathgen.py > incpath
make clean all
cd ../../cuda/src
python incpathgen.py > incpath
make clean all
cd ../../..
python setup.py build
python setup.py bdist_wheel
