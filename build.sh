#!/bin/bash

sudo apt-get install automake autoconf libtool python-dev

pushd .

cd sqaodc

git clone https://github.com/Rlovelett/eigen.git
cd eigen
git checkout 3.3.4
cd ..

git clone https://github.com/NVlabs/cub.git
cd cub
git checkout v1.7.4
cd ..

./autogen.sh
./configure --prefix=/usr --enable-cuda
make
sudo make install

popd

cd sqaodpy/
python setup.py build
python setup.py install
