#!/bin/bash

sudo apt-get install automake autoconf libtool python-dev libblas-dev

./autogen.sh
./configure --prefix=/usr --enable-cuda
make
sudo make install
