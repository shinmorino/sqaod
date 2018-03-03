#!/bin/bash

autoheader
libtoolize --force
aclocal
automake --add-missing
autoconf
