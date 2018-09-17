#!/bin/bash

docker run --rm -d -v `pwd`/..:/home/repo -p 8080:80 shinmorino/reposerv
