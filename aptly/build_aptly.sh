export GOROOT=/usr/lib/go-1.10
export PATH=${GOROOT}/bin:$PATH
export GOPATH=$HOME/go
mkdir -p $GOPATH/src/github.com/aptly-dev/aptly
git clone https://github.com/aptly-dev/aptly $GOPATH/src/github.com/aptly-dev/aptly
cd $GOPATH/src/github.com/aptly-dev/aptly
git checkout v1.3.0
make install
