# Sqaod

Collections of solvers/annealers for simulated quamtum annealing on CPU and CUDA(NVIDIA GPU).

## Project status (as of 4/1)
Alpha1 is going to be released by 4/8.<BR>
Currently apt repository is working, though python packages have not been registered to PyPI yet.<BR>

See milestones [here](https://github.com/shinmorino/sqaod/milestones?direction=asc&sort=due_date&state=open) for further development status.

## Installation  
Here's an instruction to install alpha1 binary distribution of sqaod.  Alpha1 binary distribuion is provided only for Ubuntu 16.04.<BR>
If you want to use other Linux distribution, you need to build from source. See the wiki page, [Build from source](https://github.com/shinmorino/sqaod/wiki/Build-from-source).


### 1. Installing NVIDIA Driver<BR>
If you want to run CUDA-based solvers, you need NVIDIA GPU and NVIDIA driver installed on your machine.<BR>
GPUs of compute capabiity 3.5 (2nd gen Kepler) or later is required. Recommendation is Maxwell(Compute Capability 5.0) or later.  Please visit [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) to check compute capability of GPUs.

To install CUDA packages required for sqaod, please visit [CUDA downloads](https://developer.nvidia.com/cuda-downloads), and download 'deb(network)' or 'deb(local)' package.<BR>
Afer downloading the deb package, run the following commands.  Here, CUDA 9.1, deb(network) package is assumed.
~~~
 $ sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
 $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
 $ sudo apt-get update
 $ sudo apt--get install cuda-drivers
~~~

### 2. Installing native libraries.
Sqaod has its own C++ native libraries which is invoked from python c-extensions.

~~~
 # adding sqaod repository
 $ echo 'deb [arch=amd64] https://shinmorino.github.io/sqaod/ubuntu xenial main' | \
   sudo tee /etc/apt/sources.list.d/sqaod.list
 $ apt-get update

 # install sqaodc native library.
 # if your CPU has avx2 feature, installl libsqaodc-avx2, otherwise install libsqaodc.
 $ apt-get install libsqaodc-avx2   # AVX2 enabled.
 $ apt-get install libsqaodc        # otherwise (SSE2 enabled).
 
 # install CUDA native library if you need CUDA-based solvers.
 $ apt-get install libsqaodc-cuda-9-0
~~~

**Note:** You may see a warning during installation : "W: The repository 'https://shinmorino.github.io/sqaod/ubuntu xenial Release' does not have a Release file."<BR>
If you get a confirmation like "Install these packages without verification? [y/N]", answer y to proceed installation.<BR>
It will be fixed by alpha2 release.


### 3. installing python package

Sqaod is currently supporting python 2.7 and python 3.3 - 3.5.<BR>
Use pip to install.
~~~
 $ pip install sqaod
~~~


### Running examples

Python examples are in [sqaod/sqaodpy/examples](https://github.com/shinmorino/sqaod/tree/master/sqaodpy/example).  The below is an example to run dense graph annealer.

~~~
$ wget https://github.com/shinmorino/sqaod/blob/master/sqaodpy/example/dense_graph_annealer.py
$ python dense_graph_annealer.py
~~~

### Feedback and requests
I welcome your feedback and requests.<BR>
Please file your feedback and/or requests to [Issues](https://github.com/shinmorino/sqaod/issues).<BR>

Enjoy !!!
