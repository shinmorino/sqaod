FROM nvidia/cuda:9.2-runtime-ubuntu16.04
RUN apt update && \
    apt install -y --no-install-recommends apt-transport-https curl python-pip python-setuptools ipython && \
    . /etc/lsb-release && \
    (echo "deb [arch=amd64] https://shinmorino.github.io/sqaod/ubuntu ${DISTRIB_CODENAME} multiverse" | tee /etc/apt/sources.list.d/sqaod.list) && \
    (curl -s -L https://shinmorino.github.io/sqaod/gpgkey | apt-key add -) && \
    apt update && apt install -y libsqaodc libsqaodc-dev libsqaodc-cuda-9-2 && pip install sqaod
CMD ipython
