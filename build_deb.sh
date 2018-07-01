#!/bin/bash

export CFLAGS="-O2"

rm -rf debian/*
git checkout debian
cd debian
python gen.py ${1} ${2}
cd ..

pushd .
cd ..
tar -czf sqaod_0.3.0.orig.tar.gz sqaod-0.3.0
popd
DEB_BUILD_OPTIONS='parallel=4' dpkg-buildpackage -us -uc
