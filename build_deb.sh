#!/bin/bash

# FIXME: add Eigen SIMD flags.
export CFLAGS="-O2"

rm -f debian/*.log

make clean
cd debian
cp control.gen control
cp rules.gen rules
cd ..
pushd .
cd ..
tar -czf sqaod_0.1.0.orig.tar.gz sqaod-0.1.0
popd
dpkg-buildpackage -us -uc

# FIXME: add cuda ver to package name ?
