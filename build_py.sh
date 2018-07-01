#!/bin/bash

# command line examples:
# https://packaging.python.org/guides/using-testpypi/
# pip install --index-url https://test.pypi.org/simple/ sqaod
# twine upload --repository-url https://test.pypi.org/legacy/ sqaod_py2-0.1.0-cp27-cp27mu-manylinux1.whl

rm -rf sqaodpy/build sqaodpy/dist sqaodpy/*.egg-info

cd sqaodpy/sqaod/cpu/src
python incpathgen.py > incpath
make clean all
cd ../../cuda/src
python incpathgen.py > incpath
make clean all
cd ../../..

pip install numpy==1.11
python setup.py bdist_wheel --plat-name manylinux1_x86_64 # --python-tag=py2
