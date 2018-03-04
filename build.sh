#!/bin/bash

sudo apt-get install automake autoconf libtool python-dev

pushd .

mkdir 3rdparty
cd 3rdparty

git clone https://github.com/Rlovelett/eigen.git
cd eigen
git checkout 3.3.4
cd ..

git clone https://github.com/NVlabs/cub.git
cd cub
git checkout v1.7.4
cd ..

popd

./autogen.sh
./configure --prefix=/usr --enable-cuda
make
sudo make install

cd sqaodpy/sqaod/cpu/src
python incpathgen.py > incpath
make
cd ../../cuda/src
python incpathgen.py > incpath
make
cd ../../..
python setup.py build
python setup.py bdist_wheel
