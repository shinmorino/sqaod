#!/bin/bash

mkdir -p 3rdparty
cd 3rdparty

if ! test -d eigen
then
    git clone https://github.com/Rlovelett/eigen.git
fi
cd eigen
git checkout 3.3.4
cd ..

if ! test -d cub
then
    git clone https://github.com/NVlabs/cub.git
fi
cd cub
git checkout v1.7.4
cd ..
