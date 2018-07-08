# Sqaod
#### Latest version : Beta2 0.3.0 (deb), 0.3.0 (python). 

Collections of solvers/annealers for simulated quantum annealing on CPU and CUDA(NVIDIA GPU).<BR>
Please visit [sqaod wiki](https://github.com/shinmorino/sqaod/wiki) for more details.


## Project status (as of 7/1)
### Beta2 has been released on 7/1. <BR>
 * important updates in Beta2
   - Peformance tuning for both CPU-based and CUDA-based solvers([#33](https://github.com/shinmorino/sqaod/issues/33), [#34](https://github.com/shinmorino/sqaod/issues/34)).
   - Device memory leak fixed([#51](https://github.com/shinmorino/sqaod/issues/51)).
   - BLAS disabled for better performance ([#52](https://github.com/shinmorino/sqaod/issues/52)).
 * important updates in Beta1
   - Python interface is fixed.<BR>
   API change : In alpha2, set_q() was previously used to set a bit vector and an array of bit vectors to annealers.  In beta1, set_q() is to set a bit vector, and newly-introduced set_qset() is to set an array of bit vectors.
   - SQAOD_VERBOSE env var is introduced to control log output.<BR>
   Setting SQAOD_VERBOSE as non '0' value to enable log output, otherwise logs are suppressed.
   - Stride is introduced to MatrixType<> and DeviceMatrixType<> to enable further optimizataion, which is the final library-wide modification.
   - For more details, please see [Beta1](https://github.com/shinmorino/sqaod/milestone/3) milestone.

## Installation  
Here's an instruction to install beta2 binary distribution of sqaod.  Beta2 binary distribuion is provided only for Ubuntu 16.04.<BR>
If you want to use other Linux distribution, currently you need to build from source. See wiki, [Build from source](https://github.com/shinmorino/sqaod/wiki/Build-from-source).<BR>
Or if you need a binary distribution for your linux distro, please file a request to [Issues](https://github.com/shinmorino/sqaod/issues).  Windows version and/or docker images are possible as well.

### 1. Installing NVIDIA Driver<BR>
If you want to run CUDA-based solvers, you need NVIDIA GPU and NVIDIA driver installed on your machine.<BR>
GPUs of compute capabiity 3.5 (2nd gen Kepler) or later is required. Recommendation is Maxwell(compute capability 5.0) or later.  Please visit [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) to check compute capability of GPUs.

To install CUDA packages required for sqaod, please visit [CUDA downloads](https://developer.nvidia.com/cuda-downloads), and download 'deb(network)' or 'deb(local)' package.<BR>
Afer downloading the deb package, run the following commands.  Here, CUDA 9.1, deb(network) package is assumed.
~~~
 $ sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
 $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
 $ sudo apt-get update
 $ sudo apt--get install cuda-drivers
~~~

### 2. Installing native libraries.
Sqaod has its own C++ native libraries which are invoked via python c-extensions.  These libraries are released as deb packages.  Please use apt-get to install them.

 **Note:** If you installed alpha versions (alpha1, alpha2) of libsqaod, uninstall them first, and remove apt-repository setting.
~~~
 # removing older packages if you instsalled.
 $ sudo apt-get remove libsqaod-cuda-9-0
 $ sudo apt-get remove libsqaod-avx2
 $ sudo apt-get remove libsqaod

 # remove apt-repository setting.
 $ sudo rm -f /etc/sources.list.d/sqaod.list
~~~

 For installation, please run the following.<BR>
~~~
 $ sudo apt-get update
 $ sudo apt-get install apt-transport-https apt-utils

 # adding apt repository setting.
 $ echo 'deb [arch=amd64] https://shinmorino.github.io/sqaod/ubuntu xenial multiverse' | \
   sudo tee /etc/apt/sources.list.d/sqaod.list

 # install repositry key.
 $ curl -s -L https://shinmorino.github.io/sqaod/gpgkey | sudo apt-key add -

 # update and install sqaodc native library.
 $ sudo apt-get update
 $ sudo apt-get install libsqaodc
 
 # install CUDA native library if you need CUDA-based solvers.
 $ sudo apt-get install libsqaodc-cuda-9-0
~~~


### 3. installing python package

Sqaod is currently supporting python 2.7 and python 3.3 - 3.5.  For details, please visit the project [sqaod project page](https://pypi.python.org/pypi/sqaod/) on PyPI.<BR>
To install sqaod python package, use pip as shown below.
~~~
 $ pip install -U sqaod
~~~


### 4. Running examples

Python examples are in [sqaod/sqaodpy/examples](https://github.com/shinmorino/sqaod/tree/master/sqaodpy/example).  The below is an example to run dense graph annealer.

~~~
$ wget https://raw.githubusercontent.com/shinmorino/sqaod/master/sqaodpy/example/dense_graph_annealer.py
$ python dense_graph_annealer.py
~~~

### 5. Using libraries of your choice.

 When installing libsqaodc package, sse2 and avx2 versions of native libraries are installed.  The default is the sse2 version.  To choose which version to enable, use update-alternative as shown below.

~~~
 $ sudo update-alternatives --config libsqaodc.so.0
 There are 2 choices for the alternative libsqaodc.so.0 (providing /usr/lib/libsqaodc.so.0).

   Selection    Path                                    Priority   Status
 ------------------------------------------------------------
 * 0            /usr/lib/libsqaodc-sse2/libsqaodc.so.0   50        auto mode
   1            /usr/lib/libsqaodc-avx2/libsqaodc.so.0   20        manual mode
   2            /usr/lib/libsqaodc-sse2/libsqaodc.so.0   50        manual mode
 
 Press <enter> to keep the current choice[*], or type selection number: 1
~~~

### Feedback and requests
I welcome your feedback and requests.<BR>
Please file your feedback and/or requests to [Issues](https://github.com/shinmorino/sqaod/issues).<BR>


### Opensource software used in sqaod.

- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) ([MPL2](https://www.mozilla.org/en-US/MPL/2.0/))
- [CUB](http://nvlabs.github.io/cub/) ([BSD 3-Clause "New" or "Revised" License](https://github.com/NVlabs/cub/blob/1.8.0/LICENSE.TXT))
- [libblas](https://packages.ubuntu.com/xenial/libblas3) ([Modified BSD License](http://www.netlib.org/lapack/LICENSE.txt))
- From Beta1, [aptly](https://www.aptly.info/) is used to manage sqaod repository.


### Enjoy !!!
