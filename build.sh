#!/bin/bash

sudo apt-get install automake autoconf libtool python-dev

pushd .

cd libsqaod
git clone https://github.com/Rlovelett/eigen.git
cd eigen
git checkout 3.3.4
cd ..
./autogen.sh
./configure #--prefix=/usr
make
sudo make install

popd

cd python/
python setup.py build
python setup.py install
