---
layout: post
title: "Building Tensorflow with OpenCL support on Ubuntu 16.04"
description: ""
category: 
tags: [ubuntu, tensorflow, opencl, amdgpu-pro]
---
{% include JB/setup %}

## Overview

In this post I will describe the steps required to get Tensorflow running with OpenCL on an AMD GPU.

### Install AMDGPU-PRO Drivers

Install with (applying sudo credentials where necessary):

```bash
wget -referer=https://support.amd.com https://www2.ati.com/drivers/linux/ubuntu/amdgpu-pro-17.50-511655.tar.xz
tar -xvf amdgpu-pro-17.50-511655.tar.xz
cd amdgpu-pro-17.50-511655
./amdgpupro-install --opencl=legacy --headless
sudo reboot
```

### Install a recent python

I use pyenv to manage different versions of python, setup your python as follows:

```bash
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

Add the following lines to your `~/.bashrc` file:

```bash
export PATH="~/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Restart your terminal and you have pyenv installed. Now install a python (I'll use the latest 3.6) and its system dependencies:

```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev
env PYTHON_CONFIGURE_OPTS="--enable-shared" MAKEOPTS="-j 8" pyenv install 3.6.5
pyenv local 3.6.5
pip install wheel numpy
```

### Get The ComputeCPP SYCL Implementation

Tensorflow can use the [SYCL](https://www.khronos.org/sycl) interface to seamlessly run device agnostic c++ code on an OpenCL enabled device. There is an open source template based library called [triSYCL](https://github.com/triSYCL/triSYCL). It requires c++17 support and thus you probably need to build your own gcc from a recent release. I won't cover it here but you can if you want to.

Sign up here and get your computecpp library:

[https://developer.codeplay.com/computecppce/latest/overview](https://developer.codeplay.com/computecppce/latest/overview)

```bash
tar -xvf ComputeCpp-CE-0.6.1-Ubuntu.16.04-64bit.tar.gz
sudo mv ComputeCpp-CE-0.6.1-Ubuntu-16.04-64bit /usr/local/computecpp
```

### Get Tensorflow with experimental opencl support

This fork of tensorflow is maintained by someone from Codeplay, who make ComputeCPP. It's highly experimental, expect it to change in the future. We use the dev/amd_gpu branch which is currently under active development:

```bash
git clone https://github.com/lukeiwanski/tensorflow
cd tensorflow
git fetch
git checkout dev/amd_gpu
pyenv local 3.6.5
```

In order to build tensorflow, you need to use Google's bazel build system:

```bash
wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel_0.11.1-linux-x86_64.deb
sudo dpkg -i bazel_0.11.1-linux-x86_64.deb
```

### Environment and Building

You can now configure and build tensorflow, from the tensorflow directory run:

```bash
./configure
```

I used the following options (press enter to get the defaults):

```bash
$ ./configure 
You have bazel 0.11.1 installed.
Please specify the location of python. [Default is /home/user/.pyenv/versions/3.6.5/bin/python]: 

Please input the desired Python library path to use.  Default is [/home/user/.pyenv/versions/3.6.5/lib/python3.6/site-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: Y
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [y/N]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: y
OpenCL SYCL support will be enabled for TensorFlow.

Please specify which C++ compiler should be used as the host C++ compiler. [Default is /usr/bin/g++]: 


Please specify which C compiler should be used as the hostC compiler. [Default is /usr/bin/gcc]: 


Do you wish to build TensorFlow with ComputeCPP support? [Y/n]: Y
ComputeCPP support will be enabled for TensorFlow.

Please specify the location where ComputeCpp for SYCL 1.2 is installed. [Default is /usr/local/computecpp]: 

Do you wish to build TensorFlow with double types in SYCL support? [Y/n]: Y
double types in SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with half types in SYCL support? [y/N]: n
No half types in SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:

```

Now you can build the package, consider going to get lunch because this will take a while:

```bash
bazel build --config=opt --config=sycl //tensorflow/tools/pip_package:build_pip_package
```

Hopefully it built without errors, if it did, try to fix them. Now install the generated wheel  file:

```bash
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tf_cpp_whl
pip install /tmp/tf_cpp_whl/tensorflow-1.6.0rc0-cp27-cp27mu-linux_x86_64.whl
```

Now you've got tensorflow installed, but it's not over yet, you need everything setup nicely to point to the appropriate OpenCL ICD and use everything in your updated amdgpu-pro drivers. Consider putting these environment variables in your `~/.bashrc` file:

```bash
export $OPENCL_VENDOR_PATH=/opt/amdgpu-pro/etc/OpenCL/vendors/
mkdir -p $OPENCL_VENDOR_PATH
echo "/opt/amdgpu-pro/lib/x86_64-linux-gnu/libamdocl64.so" > $OPENCL_VENDOR_PATH/amdocl64.icd
export LD_LIBRARY_PATH=/usr/local/computecpp/lib:/opt/amdgpu-pro/lib/x86_64-linux-gnu
export PATH=/usr/local/computecpp/bin:/opt/amdgpu-pro/bin:$PATH
```


Here's some explanation:

1. `OPENCL_VENDOR_PATH` is where libOpenCL.so will look for `.icd` files which point to a specific library for a driver. In our case it's the `libamdocl64.so`, each one will appear as a different vendor, so you could install [pocl](https://github.com/pocl/pocl) or [oclgrind](https://github.com/jrprice/Oclgrind)
2. `LD_LIBRARY_PATH` is where the system looks for shared object files
3. `PATH` is where the system looks like for executable files. The two listed above contain clinfo and computecpp_info

Run `clinfo` and you should get something like this describing your GPU(s):

```
Number of platforms:				 1
  Platform Profile:				 FULL_PROFILE
  Platform Version:				 OpenCL 2.1 AMD-APP (2527.3)
  Platform Name:				 AMD Accelerated Parallel Processing
  Platform Vendor:				 Advanced Micro Devices, Inc.
  Platform Extensions:				 cl_khr_icd cl_amd_event_callback cl_amd_offline_devices 


  Platform Name:				 AMD Accelerated Parallel Processing
Number of devices:				 1
  Device Type:					 CL_DEVICE_TYPE_GPU
  Vendor ID:					 1002h
  Board name:					 AMD Radeon HD 8500 Series
  Device Topology:				 PCI[ B#1, D#0, F#0 ]
  Max compute units:				 6
  Max work items dimensions:			 3
    Max work items[0]:				 1024
    Max work items[1]:				 1024
    Max work items[2]:				 1024
  Max work group size:				 256
  Preferred vector width char:			 4
  Preferred vector width short:			 2
  Preferred vector width int:			 1
  Preferred vector width long:			 1
  Preferred vector width float:			 1
  Preferred vector width double:		 1
  Native vector width char:			 4
  Native vector width short:			 2
  Native vector width int:			 1
  Native vector width long:			 1
  Native vector width float:			 1
  Native vector width double:			 1
  Max clock frequency:				 780Mhz
  Address bits:					 64
  Max memory allocation:			 330696294
  Image support:				 Yes
  Max number of images read arguments:		 128
  Max number of images write arguments:		 8
  Max image 2D width:				 16384
  Max image 2D height:				 16384
  Max image 3D width:				 2048
  Max image 3D height:				 2048
  Max image 3D depth:				 2048
  Max samplers within kernel:			 16
  Max size of kernel argument:			 1024
  Alignment (bits) of base address:		 2048
  Minimum alignment (bytes) for any datatype:	 128
  Single precision floating point capability
    Denorms:					 No
    Quiet NaNs:					 Yes
    Round to nearest even:			 Yes
    Round to zero:				 Yes
    Round to +ve and infinity:			 Yes
    IEEE754-2008 fused multiply-add:		 Yes
  Cache type:					 Read/Write
  Cache line size:				 64
  Cache size:					 16384
  Global memory size:				 505212928
  Constant buffer size:				 65536
  Max number of constant args:			 8
  Local memory type:				 Scratchpad
  Local memory size:				 32768
  Max pipe arguments:				 0
  Max pipe active reservations:			 0
  Max pipe packet size:				 0
  Max global variable size:			 0
  Max global variable preferred total size:	 0
  Max read/write image args:			 0
  Max on device events:				 0
  Queue on device max size:			 0
  Max on device queues:				 0
  Queue on device preferred size:		 0
  SVM capabilities:
    Coarse grain buffer:			 No
    Fine grain buffer:				 No
    Fine grain system:				 No
    Atomics:					 No
  Preferred platform atomic alignment:		 0
  Preferred global atomic alignment:		 0
  Preferred local atomic alignment:		 0
  Kernel Preferred work group size multiple:	 64
  Error correction support:			 0
  Unified memory for Host and Device:		 0
  Profiling timer resolution:			 1
  Device endianess:				 Little
  Available:					 Yes
  Compiler available:				 Yes
  Execution capabilities:
    Execute OpenCL kernels:			 Yes
    Execute native function:			 No
  Queue on Host properties:
    Out-of-Order:				 No
    Profiling :					 Yes
  Queue on Device properties:
    Out-of-Order:				 No
    Profiling :					 No
  Platform ID:					 0x7f9728fbf510
  Name:						 Oland
  Vendor:					 Advanced Micro Devices, Inc.
  Device OpenCL C version:			 OpenCL C 1.2 
  Driver version:				 2527.3
  Profile:					 FULL_PROFILE
  Version:					 OpenCL 1.2 AMD-APP (2527.3)
  Extensions:					 cl_khr_fp64 cl_amd_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_gl_sharing cl_amd_device_attribute_query cl_amd_vec3 cl_amd_printf cl_amd_media_ops cl_amd_media_ops2 cl_amd_popcnt cl_khr_image2d_from_buffer cl_khr_spir cl_khr_gl_event 

```

Running `computecpp_info` should get you something like this:

```
********************************************************************************

ComputeCpp Info (CE 0.6.1)

********************************************************************************

Toolchain information:

GLIBC version: 2.23
GLIBCXX: 20160609
This version of libstdc++ is supported.

********************************************************************************


Device Info:

Discovered 1 devices matching:
  platform    : <any>
  device type : <any>

--------------------------------------------------------------------------------
Device 0:

  Device is supported                     : UNTESTED - Vendor not tested on this OS
  CL_DEVICE_NAME                          : Oland
  CL_DEVICE_VENDOR                        : Advanced Micro Devices, Inc.
  CL_DRIVER_VERSION                       : 2527.3
  CL_DEVICE_TYPE                          : CL_DEVICE_TYPE_GPU 

If you encounter problems when using any of these OpenCL devices, please consult
this website for known issues:
https://computecpp.codeplay.com/releases/v0.6.1/platform-support-notes

********************************************************************************
```

### Verification
You can now quickly check that everything installed correctly (don't run this from the tensorflow source directory, you'll run into issues):

```bash
cd ~
pyenv local 3.6.5
python -c "from tensorflow.python.client import device_lib;device_lib.list_local_devices()"
2018-04-04 16:39:13.673362: I ./tensorflow/core/common_runtime/sycl/sycl_device.h:70] Found following OpenCL devices:
2018-04-04 16:39:13.673394: I ./tensorflow/core/common_runtime/sycl/sycl_device.h:72] id: 0, type: GPU, name: Oland, vendor: Advanced Micro Devices, Inc., profile: FULL_PROFILE
```

Now you should be able to run some code that uses the AMD GPU! Here's a small example using keras (which you need to install):

```bash
pip install keras
```

Here's a toy regression example (it learns the sin function) that you can use to compare the speed of CPU vs. GPU:

```python
#!/usr/bin/python
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


x = np.arange(10000)
y = np.sin(x)
for i in  ('/cpu:0', '/gpu:0'):
    with tf.device(i):
        model = Sequential([
            Dense(128, input_shape=(1,), activation='relu'),
            Dense(1),
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(x,y, validation_split=0.25, epochs=10)

```
Running this I get the following output:

```
Using TensorFlow backend.
Train on 7500 samples, validate on 2500 samples
Epoch 1/10
2018-04-04 16:53:55.766232: I ./tensorflow/core/common_runtime/sycl/sycl_device.h:70] Found following OpenCL devices:
2018-04-04 16:53:55.766265: I ./tensorflow/core/common_runtime/sycl/sycl_device.h:72] id: 0, type: GPU, name: Oland, vendor: Advanced Micro Devices, Inc., profile: FULL_PROFILE
7500/7500 [==============================] - 0s 53us/step - loss: 29006.8066 - val_loss: 0.5027
Epoch 2/10
7500/7500 [==============================] - 0s 25us/step - loss: 0.5071 - val_loss: 0.5019
Epoch 3/10
7500/7500 [==============================] - 0s 26us/step - loss: 0.5091 - val_loss: 0.5005
Epoch 4/10
7500/7500 [==============================] - 0s 25us/step - loss: 0.5105 - val_loss: 0.5007
Epoch 5/10
7500/7500 [==============================] - 0s 30us/step - loss: 0.5166 - val_loss: 0.5006
Epoch 6/10
7500/7500 [==============================] - 0s 27us/step - loss: 0.5141 - val_loss: 0.6930
Epoch 7/10
7500/7500 [==============================] - 0s 26us/step - loss: 0.5173 - val_loss: 0.5968
Epoch 8/10
7500/7500 [==============================] - 0s 27us/step - loss: 0.5162 - val_loss: 0.5382
Epoch 9/10
7500/7500 [==============================] - 0s 26us/step - loss: 0.5195 - val_loss: 0.5377
Epoch 10/10
7500/7500 [==============================] - 0s 25us/step - loss: 0.5160 - val_loss: 0.5020

Train on 7500 samples, validate on 2500 samples
Epoch 1/10
7500/7500 [==============================] - 92s 12ms/step - loss: nan - val_loss: nan
Epoch 2/10
7500/7500 [==============================] - 2s 265us/step - loss: nan - val_loss: nan
Epoch 3/10
7500/7500 [==============================] - 2s 262us/step - loss: nan - val_loss: nan
Epoch 4/10
7500/7500 [==============================] - 2s 268us/step - loss: nan - val_loss: nan
Epoch 5/10
7500/7500 [==============================] - 2s 262us/step - loss: nan - val_loss: nan
Epoch 6/10
7500/7500 [==============================] - 2s 268us/step - loss: nan - val_loss: nan
Epoch 7/10
7500/7500 [==============================] - 2s 268us/step - loss: nan - val_loss: nan
Epoch 8/10
7500/7500 [==============================] - 2s 259us/step - loss: nan - val_loss: nan
Epoch 9/10
7500/7500 [==============================] - 2s 258us/step - loss: nan - val_loss: nan
Epoch 10/10
7500/7500 [==============================] - 2s 264us/step - loss: nan - val_loss: nan

```

As you can see, in the GPU run, it takes a long time to run the first epoch of the GPU the model, and significantly longer to run each training step (also that the GPU learnt nothing). You can play with the code to see if you can get it to work but waiting ~90s for the model to start the first epoch is a pain for testing.

It might just be I have an old workstation card, or that there is a lot of overhead involved in using SYCL, I hope this helps someone else!


