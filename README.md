# Collective Knowledge workflow for image classification submitted to [ReQuEST at ASPLOS'18](http://cknowledge.org/request-cfp-asplos2018.html)

* **Title:** Optimizing [MobileNets-v1](https://arxiv.org/pdf/1704.04861.pdf) on Arm Mali GPUs
* **Authors:** Nikolay Chunosov, Flavio Vella, Anton Lokhmotov, Grigori Fursin
* **License:** [Collective Knowledge](https://github.com/ctuning/ck/blob/master/LICENSE.txt) (3-clause BSD)

## Artifact check-list (meta-information)

We use the standard [Artifact Description check-list](http://ctuning.org/ae/submission_extra.html) from systems conferences including CGO, PPoPP, PACT and SuperComputing.

* **Algorithm:** image classification
* **Program:** [MobileNets-v1](https://github.com/tensorflow/models/blob/1630da3434974e9ad5a0b6d887ac716a97ce03d3/research/slim/nets/mobilenet_v1.md#pre-trained-models) using [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) v17.12+ and [TensorFlow](https://github.com/tensorflow/tensorflow) v1.7.0+
* **Compilation:** GCC v6+ (recommended v7+); Python 2.7+ or 3.4+
* **Transformations:**
* **Binary:** compiled from source on a target platform
* **Data set:** [ImageNet](http://www.image-net.org) 2012 validation (50,000 images)
* **Run-time environment:** Linux; OpenCL v1.2+
* **Hardware:** [Linaro HiKey960](https://www.96boards.org/product/hikey960/), [Firefly RK3399](http://en.t-firefly.com/product/rk3399.html) development boards (or similar)
* **Run-time state:** 
* **Execution:** CPU and GPU frequencies set to the maximum
* **Metrics:** total execution time; top1/top5 accuracy over some (all) images from the data set
* **Output:** classification result; execution time; accuracy
* **Experiments:** CK command line
* **How much disk space required (approximately)?** TBC
* **How much time is needed to prepare workflow (approximately)?** 20 minutes
* **How much time is needed to complete experiments (approximately)?**
* **Collective Knowledge workflow framework used?** Yes
* **Publicly available?:** Yes
* **Experimental results:** https://github.com/ctuning/ck-request-asplos18-results-mobilenets-armcl-opencl

## Installation

### Install global prerequisites (Ubuntu)

**NB:** Execute commands prefixed with `#` sign under `root` or using `sudo`.

```
# apt install python python-pip
# apt install libblas-dev liblapack-dev libatlas-base-dev python-numpy python-scipy
# pip install pillow
```

### Install Collective Knowledge

```
# pip install ck
```

### Get submission repository

```
$ ck pull repo --url=https://github.com/dividiti/ck-request-asplos18-mobilenets-armcl-opencl
```

### Detect and test OpenCL driver

```
$ ck detect platform.gpgpu --opencl
```

When you run CK for the very first time, you may be asked 
to select to most close CK platform description 
shared by CK users. In our case it should be *hikey960-linux*. 

CK workflows will then use various platform-specific scripts 
from the *platform.init:hikey960-linux* such as monitoring 
or setting up CPU and GPU frequency:
```
$ ls `ck find platform.init:hikey960-linux`
```

You can later change it as following:
```
$ ck ls platform.init | sort
$ ck detect platform.os --update_platform_init \
  --platform_init_uoa={one of above CK entries}
```

### Pre-install CK dependencies

**NB:** We suggest to pre-install the following dependencies to test the workflows:

```
$ ck install ck-caffe:package:imagenet-2012-aux
$ ck install ck-caffe:package:imagenet-2012-val-min-resized
$ ck install ck-math:package:lib-armcl-opencl-18.03 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-1.0-224-npy
```

### Build and make a sample run

```
$ ck compile ck-request-asplos18-mobilenets-armcl-opencl:program:mobilenets-armcl-opencl
$ ck run ck-request-asplos18-mobilenets-armcl-opencl:program:mobilenets-armcl-opencl
...
--------------------------------
Process results in predictions
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.73 - (65) n01751748 sea snake
0.07 - (67) n01755581 diamondback, diamondback rattlesnake, Cr...
0.06 - (53) n01728920 ringneck snake, ring-necked snake, ring ...
0.04 - (60) n01740131 night snake, Hypsiglena torquata
0.03 - (54) n01729322 hognose snake, puff adder, sand viper
---------------------------------------
Accuracy top 1: 1.000000 (1 of 1)
Accuracy top 5: 1.000000 (1 of 1)
--------------------------------


  (reading fine grain timers from tmp-ck-timer.json ...)

{
  "accuracy_top1": 1.0, 
  "accuracy_top5": 1.0, 
  "execution_time": 0.293805, 
  "execution_time_sum": 1.9365359999999998, 
  "frame_predictions": [
    {
      "accuracy_top1": "yes", 
      "accuracy_top5": "yes", 
      "class_correct": 65, 
      "class_topmost": 65, 
      "file_name": "ILSVRC2012_val_00000001.JPEG"
    }
  ], 
  "images_load_time_avg_s": 0.017385, 
  "images_load_time_s": 0.017385, 
  "prediction_time_avg_s": 0.293805, 
  "prediction_time_total_s": 0.293805, 
  "setup_time_s": 1.625346, 
  "test_time_s ": 0.321721
}
```

## Exploring performance and accuracy of MobileNets using Arm Compute Library (ArmCL)

### Install ArmCL variants

```
$ ck install ck-math:package:lib-armcl-opencl-17.12 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install ck-math:package:lib-armcl-opencl-18.01 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install ck-math:package:lib-armcl-opencl-18.03 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install package:lib-armcl-opencl-request
```

**NB:** It is necessary to specify `--env.USE_GRAPH=ON --env.USE_NEON=ON` for packages from `repo:ck-math`, but not for `package:lib-armcl-opencl-request`.

### Install MobileNets weights

To install all the weights in one go:
```
$ ck install package --tags=mobilenet-v1-all,npy
$ cd $CK_TOOLS && du -hcs weights-mobilenet-v1_*
2.4M    weights-mobilenet-v1_0.25_128-npy
2.4M    weights-mobilenet-v1_0.25_160-npy
2.4M    weights-mobilenet-v1_0.25_192-npy
2.4M    weights-mobilenet-v1_0.25_224-npy
5.6M    weights-mobilenet-v1_0.50_128-npy
5.6M    weights-mobilenet-v1_0.50_160-npy
5.6M    weights-mobilenet-v1_0.50_192-npy
5.6M    weights-mobilenet-v1_0.50_224-npy
11M     weights-mobilenet-v1_0.75_128-npy
11M     weights-mobilenet-v1_0.75_160-npy
11M     weights-mobilenet-v1_0.75_192-npy
11M     weights-mobilenet-v1_0.75_224-npy
17M     weights-mobilenet-v1_1.0_128-npy
17M     weights-mobilenet-v1_1.0_160-npy
17M     weights-mobilenet-v1_1.0_192-npy
17M     weights-mobilenet-v1_1.0_224-npy
140M    total
```

Alternatively, install the weights individually as below.

To install the weights for images with resolution `224x224`:
```
$ ck install package:weights-mobilenet-v1-1.0-224-npy
$ ck install package:weights-mobilenet-v1-0.75-224-npy
$ ck install package:weights-mobilenet-v1-0.50-224-npy
$ ck install package:weights-mobilenet-v1-0.25-224-npy
```

To install the weights for images with resolution `192x192`:
```
$ ck install package:weights-mobilenet-v1-1.0-192-npy
$ ck install package:weights-mobilenet-v1-0.75-192-npy
$ ck install package:weights-mobilenet-v1-0.50-192-npy
$ ck install package:weights-mobilenet-v1-0.25-192-npy
```

To install the weights for images with resolution `160x160`:
```
$ ck install package:weights-mobilenet-v1-1.0-160-npy
$ ck install package:weights-mobilenet-v1-0.75-160-npy
$ ck install package:weights-mobilenet-v1-0.50-160-npy
$ ck install package:weights-mobilenet-v1-0.25-160-npy
```

To install the weights for images with resolution `128x128`:
```
$ ck install package:weights-mobilenet-v1-1.0-128-npy
$ ck install package:weights-mobilenet-v1-0.75-128-npy
$ ck install package:weights-mobilenet-v1-0.50-128-npy
$ ck install package:weights-mobilenet-v1-0.25-128-npy
```

To check all the installed weights:
```
$ ck show env --tags=mobilenet,weights,npy
```

### Make a sample run
To test a scaled MobileNet architecture, please specify:
 - **resolution** : 224 (default), 192, 160, 128;
 - **width_multiplier** : 1.0 (default), 0.75, 0.5, 0.25.

```
$ ck benchmark program:mobilenets-armcl-opencl \
  --env.CK_ENV_MOBILENET_RESOLUTION=192 \
  --env.CK_ENV_MOBILENET_WIDTH_MULTIPLIER=0.75
```
Then, select the desired ArmCL and MobileNets variants.

### Performance evaluation of the MobileNets family
```
$ cd `ck find script:mobilenets-armcl-opencl`
$ python benchmark.py --repetitions=3
```

### Accuracy evaluation of the MobileNets family
```
$ cd `ck find script:mobilenets-armcl-opencl`
$ python benchmark.py --repetitions=1 --accuracy
```

### Check the experimental data
```
$ ck list local:experiment:*
 or
$ ck search experiment --tags=request-asplos18
 or
$ ck search experiment --tags=request-asplos18,performance
```

If something goes wrong, you can remove experimental results 
before starting new exploration as following:

```
$ ck rm experiment:* --tags=request-asplos18
 or
$ ck rm experiment:* --tags=request-asplos18 --force

```

You can also pack all experimental results to share with colleagues
```
$ ck zip local:experiment:*
```

CK will create a "ckr-local.zip" file with CK entries.

You can then unzip it to the local (or other) repository on another machine via CK as following:
```
$ ck unzip repo:local --zip=ckr-local.zip
```

## Shared experimental results 

We shared raw experimental results in the CK format 
in this [CK repo](https://github.com/ctuning/ck-request-asplos18-results-mobilenets-armcl-opencl).

You can view and test them as following:
```
$ ck pull repo:ck-request-asplos18-results-mobilenets-armcl-opencl
$ ck ls ck-request-asplos18-results-mobilenets-armcl-opencl:experiment:* | sort
$ ck dashboard request.apslos18
```

## Unify output and add extra dimensions

Scripts to unify all experiments and add extra dimensions in ReQuEST format for further comparison and visualization are available in the following entry:
```
$ cd `ck find ck-request-asplos18-mobilenets-armcl-opencl:script:mobilenets-armcl-opencl`
```

- benchmark-merge-performance-with-accuracy.py - merges performance entries with accuracy
- benchmark-add-dimensions.py - adds extra dimensions

All updated experimental results are then moved to [ck-request-asplos18-results-mobilenets-armcl-opencl repository](https://github.com/ctuning/ck-request-asplos18-results-mobilenets-armcl-opencl).
The best configurations are also moved to [ck-request-asplos18-results repo](https://github.com/ctuning/ck-request-asplos18-results).


## Exploring performance and accuracy of MobileNets using TensorFlow

### Install TensorFlow dependencies

```
# apt install liblapack-dev libatlas-dev
# pip install enum34 mock pillow wheel absl-py scipy
```

### Install CK-TensorFlow

```
$ ck pull repo:ck-tensorflow
$ ck install ck-env:package:tool-bazel-0.11.1-linux
$ ck install package:lib-tensorflow-1.7.0-src-cpu [--env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1]
```

**NB:** You may want to restrict the number of build threads to 1 or 2 on a platform with less than 4 GB RAM. For example, add `--env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=2` on HiKey960 (3 GB RAM with swap enabled) or `--env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1` on Tegra TX1 (4 GB RAM without swap enabled).

### Install MobileNets weights

To install all the pretrained MobileNets-v1 weights shared in 2017 (which were used for this evaluation):
```
$ ck install package --tags=mobilenet-v1-all,tensorflowmodel,2017_06_14
$ cd $CK_TOOLS && du -hsc tensorflowmodel-mobilenet-v1-*-py
12M     tensorflowmodel-mobilenet-v1-0.25-128-py
12M     tensorflowmodel-mobilenet-v1-0.25-160-py
12M     tensorflowmodel-mobilenet-v1-0.25-192-py
12M     tensorflowmodel-mobilenet-v1-0.25-224-py
25M     tensorflowmodel-mobilenet-v1-0.50-128-py
25M     tensorflowmodel-mobilenet-v1-0.50-160-py
25M     tensorflowmodel-mobilenet-v1-0.50-192-py
25M     tensorflowmodel-mobilenet-v1-0.50-224-py
44M     tensorflowmodel-mobilenet-v1-0.75-128-py
44M     tensorflowmodel-mobilenet-v1-0.75-160-py
44M     tensorflowmodel-mobilenet-v1-0.75-192-py
44M     tensorflowmodel-mobilenet-v1-0.75-224-py
69M     tensorflowmodel-mobilenet-v1-1.0-128-py
69M     tensorflowmodel-mobilenet-v1-1.0-160-py
69M     tensorflowmodel-mobilenet-v1-1.0-192-py
69M     tensorflowmodel-mobilenet-v1-1.0-224-py
595M    total
```

To install the pretrained MobileNets-v1 weights shared in 2018:
```
$ ck install package --tags=mobilenet-v1-all,tensorflowmodel,2018_02_22
$ cd $CK_TOOLS && du -hsc tensorflowmodel-mobilenet-v1-*-2018_02_22-py
11M     tensorflowmodel-mobilenet-v1-0.25-128-2018_02_22-py
11M     tensorflowmodel-mobilenet-v1-0.25-160-2018_02_22-py
11M     tensorflowmodel-mobilenet-v1-0.25-192-2018_02_22-py
11M     tensorflowmodel-mobilenet-v1-0.25-224-2018_02_22-py
35M     tensorflowmodel-mobilenet-v1-0.50-128-2018_02_22-py
35M     tensorflowmodel-mobilenet-v1-0.50-160-2018_02_22-py
35M     tensorflowmodel-mobilenet-v1-0.50-192-2018_02_22-py
35M     tensorflowmodel-mobilenet-v1-0.50-224-2018_02_22-py
43M     tensorflowmodel-mobilenet-v1-0.75-128-2018_02_22-py
43M     tensorflowmodel-mobilenet-v1-0.75-160-2018_02_22-py
43M     tensorflowmodel-mobilenet-v1-0.75-192-2018_02_22-py
43M     tensorflowmodel-mobilenet-v1-0.75-224-2018_02_22-py
69M     tensorflowmodel-mobilenet-v1-1.0-128-2018_02_22-py
69M     tensorflowmodel-mobilenet-v1-1.0-160-2018_02_22-py
69M     tensorflowmodel-mobilenet-v1-1.0-192-2018_02_22-py
69M     tensorflowmodel-mobilenet-v1-1.0-224-2018_02_22-py
624M    total
```
