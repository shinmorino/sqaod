#!/bin/bash

aptly -config=aptly.conf publish repo sqaod_xenial filesystem:root:ubuntu
aptly -config=aptly.conf publish repo sqaod_artful filesystem:root:ubuntu


