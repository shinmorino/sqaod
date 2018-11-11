# Sqaod
#### Latest version : v1.0.2 (deb), 1.0.2 (python). (Nov. 11, 2018)

Collections of solvers/annealers for simulated quantum annealing on CPU and CUDA(NVIDIA GPU).<BR>
Please visit [sqaod wiki](https://github.com/shinmorino/sqaod/wiki) for more details.

## Project status
### Version 1.0.2 Released (2018/11/11)
Version 1.0.2 includes miscellaneous bug fixes that affect annealing behavior.
Please update to 1.0.2 if you're using older versions.

- getSystemE() is added to solvers to calculate system energy during annealing. [[#60]](https://github.com/shinmorino/sqaod/issues/60)
- sqaod.algorithm.sa_default is added to select default SA algorithms in annealers. [[#61]](https://github.com/shinmorino/sqaod/issues/61)
- calculate_E() and make_solutions() are not required to get QUBO energy and solutions.  These functions are for caching energies and solutions. [[#63]](https://github.com/shinmorino/sqaod/issues/63)
- Python solvers return copies of objects.[[#62]](https://github.com/shinmorino/sqaod/issues/62)
- Fix: anneal_one_step() for SA algorithm did not work, since parameters are not correctly passed. [[#65]](https://github.com/shinmorino/sqaod/issues/65)
- Fix: QUBO energy was not correctly calculated and beta was not correctly applied in SQA algorithms. [[#64]](https://github.com/shinmorino/sqaod/issues/64)
- Fix: symmetrize() was not correctly handled. [[#66]](https://github.com/shinmorino/sqaod/issues/66)

Please visit the '[Release history](https://github.com/shinmorino/sqaod/wiki/Release-history)' page for changes and updates.

#### Future plan
- Version 1.1 planning is undergoing.  Please file your requests to [Version 1.1 planning [#55]]( https://github.com/shinmorino/sqaod/issues/55).

## Installation  

If you're using Ubuntu 16.04/18.04 or CentOS(RHEL) 7, please visit [Installation](https://github.com/shinmorino/sqaod/wiki/Installation) page at sqaod wiki.

If you want to use other Linux distribution, you need to build from source. See wiki, [Build from source](https://github.com/shinmorino/sqaod/wiki/Build-from-source).
Please file a request to [Issues](https://github.com/shinmorino/sqaod/issues) if you need binary distribution for your linux distro.  Windows version and/or docker images are possible as well.


Here's a quick instruction to install sqaod v1.0 to Ubuntu 16.04/18.04.


### 1. Cleaning environment.

If you installed sqaod from source, clean up native libraries at first.
~~~
find /usr/lib | grep libsqaodc
# if you find libraries, remove them.
find /usr/lib | grep libsqaodc | sudo xargs rm -f
~~~

If you installed alpha versions (alpha1, alpha2) of libsqaod, uninstall them first, and remove apt-repository setting.
~~~
# removing older packages if you instsalled.
sudo apt-get remove libsqaodc-cuda-9-0
sudo apt-get remove libsqaodc-avx2
sudo apt-get remove libsqaodc

# remove apt-repository setting.
sudo rm -f /etc/sources.list.d/sqaod.list
~~~


### 2. Installing native libraries

~~~
sudo apt-get update
sudo apt-get install apt-transport-https apt-utils

# adding apt repository setting.
 
. /etc/lsb-release
echo "deb [arch=amd64] https://shinmorino.github.io/sqaod/ubuntu ${DISTRIB_CODENAME} multiverse" | \
   sudo tee /etc/apt/sources.list.d/sqaod.list

# install repository key.
curl -s -L https://shinmorino.github.io/sqaod/gpgkey | sudo apt-key add -

# update and install sqaodc native library.
sudo apt-get update
sudo apt-get install libsqaodc
~~~

### 3. Installing CUDA driver/libraries (if you need CUDA-based solvers.)

~~~
distribution=$(. /etc/os-release;echo $ID${VERSION_ID//.})
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/${distribution}/x86_64 /" | \
   sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/${distribution}/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-drivers

sudo apt-get install libsqaodc-cuda-10-0

# If you prefer CUDA 9.0, run following. (Packages with CUDA 9.0 is available on Ubuntu16.04.)
sudo apt-get install libsqaodc-cuda-9-0

~~~

### 4. Installing python package

#### Python 2.7/3.5/3.6/3.7.
To install sqaod python package, use pip as shown below.
~~~
pip install -U sqaod
~~~

### 5. Running examples

Python examples are in [sqaod/sqaodpy/examples](https://github.com/shinmorino/sqaod/tree/master/sqaodpy/example).  The below is an example to run dense graph annealer.

~~~
curl -s -L -O https://raw.githubusercontent.com/shinmorino/sqaod/master/sqaodpy/example/dense_graph_annealer.py
python dense_graph_annealer.py
~~~

### Dockerfile
[Dockerfile](https://github.com/shinmorino/sqaod/tree/gh-pages/docker/kato) for Ubuntu-16.04 with CUDA-9.2, contribution from [Kato-san](https://github.com/gyu-don).


### Feedback and requests
I welcome your feedback and requests.<BR>
Please file your feedback and/or requests to [Issues](https://github.com/shinmorino/sqaod/issues).<BR>


### Opensource software used in sqaod.

- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) ([MPL2](https://www.mozilla.org/en-US/MPL/2.0/))
- [CUB](http://nvlabs.github.io/cub/) ([BSD 3-Clause "New" or "Revised" License](https://github.com/NVlabs/cub/blob/1.8.0/LICENSE.TXT))
- [libblas](https://packages.ubuntu.com/xenial/libblas3) ([Modified BSD License](http://www.netlib.org/lapack/LICENSE.txt))
- From Beta1, [aptly](https://www.aptly.info/) is used to manage sqaod repository.


### Enjoy !!!
