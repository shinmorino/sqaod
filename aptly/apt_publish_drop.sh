#!/bin/bash

aptly -config=aptly.conf publish drop xenial filesystem:root:ubuntu
aptly -config=aptly.conf publish drop artful filesystem:root:ubuntu
