#!/bin/bash

make clean
pushd .
cd ..
tar -czf sqaod-0.1.0.tar.gz sqaod-0.1.0
popd
dpkg-buildpackage -us -uc

# FIXME: add cuda ver to package name ?
