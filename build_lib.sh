#!/bin/bash

sudo apt-get install -y git
./clone_3rdparty.sh

sudo apt-get -y install automake autoconf libtool python-dev libblas-dev

./autogen.sh
./configure --prefix=/usr --enable-cuda
make
sudo make install
