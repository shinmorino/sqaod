#!/bin/bash

mkdir -p 3rdparty
cd 3rdparty

if ! test -d eigen
then
    git clone https://github.com/eigenteam/eigen-git-mirror.git eigen
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
