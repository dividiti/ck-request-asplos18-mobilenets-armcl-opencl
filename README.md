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
* **Scoreboard:** http://cKnowledge.org/request-results

## Installation

### Install global prerequisites (Ubuntu)

**NB:** Execute commands prefixed with `#` sign under `root` or using `sudo`.

```
# apt install libblas-dev liblapack-dev libatlas-base-dev
# apt install python python-pip python-numpy python-scipy
# pip install pillow
```

### Minimal CK installation

The minimal installation requires:

* Python 2.7 or 3.3+ (limitation is mainly due to unitests)
* Git command line client.

You can install CK in your local user space as follows:

```
$ git clone http://github.com/ctuning/ck
$ export PATH=$PWD/ck/bin:$PATH
$ export PYTHONPATH=$PWD/ck:$PYTHONPATH
```

You can also install CK via PIP with sudo to avoid setting up environment variables yourself:

```
$ sudo pip install ck
```

### Install CK repositories and a sample dataset

```
$ ck pull repo --url=https://github.com/dividiti/ck-request-asplos18-mobilenets-armcl-opencl
$ ck install ck-caffe:package:imagenet-2012-val-min-resized
$ ck install ck-caffe:package:imagenet-2012-aux
```

### Detect and test OpenCL driver

```
$ ck detect platform.gpgpu --opencl
```

If you are prompted to choose a platform description, select the one the name of which is the same or similar to your platform (e.g. `hikey960-linux`) or `generic-linux`.

You can later change it as follows:
```
$ ck ls platform.init | sort
$ ck detect platform.os --update_platform_init --platform_init_uoa=<one of the listed CK entries>
```

For example, for HiKey960 choose `hikey960-linux`, for RK3399 choose `firefly-linux`.

CK workflows will then use various platform-specific scripts such as for monitoring or setting up the CPU and GPU frequencies:
```
$ ls `ck find platform.init:hikey960-linux`
```


## Exploring performance and accuracy of MobileNets using the Arm Compute Library (ArmCL)

### Install ArmCL variants

Install [dividiti](http://dividiti.com)'s fork of [ArmCL 18.03](https://github.com/ARM-software/ComputeLibrary/releases/tag/v18.03) with a [new direct convolution kernel](https://github.com/ARM-software/ComputeLibrary/pull/432):

```
$ ck install package:lib-armcl-opencl-request
```

Optionally, install one or more of the official ArmCL releases:

```
$ ck install ck-math:package:lib-armcl-opencl-18.03 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install ck-math:package:lib-armcl-opencl-18.01 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install ck-math:package:lib-armcl-opencl-17.12 --env.USE_GRAPH=ON --env.USE_NEON=ON
```

**NB:** It is necessary to specify `--env.USE_GRAPH=ON --env.USE_NEON=ON` for the official release packages (from `repo:ck-math`), but not for `package:lib-armcl-opencl-request`.

To check all the installed ArmCL variants:
```
$ ck show env --tags=armcl
Env UID:         Target OS: Bits: Name:                         Version:         Tags:

fbf64d81a58d1c44   linux-64    64 ARM Compute Library (request) request-d8f69c13 64bits,arm,arm-compute-library,armcl,host-os-linux-64,lib,target-os-linux-64,v0,v0.0,vgraph,vneon,vopencl,vrequest
0241079df9c64221   linux-64    64 ARM Compute Library (opencl)  18.03-e40997bb   64bits,arm,arm-compute-library,armcl,channel-stable,host-os-linux-64,lib,target-os-linux-64,v18,v18.03,v18.3,v18.3.0,vdefault,vgraph,vneon,vopencl
8d8a8e65584dac8e   linux-64    64 ARM Compute Library (opencl)  18.01-f45d5a9b   64bits,arm,arm-compute-library,armcl,channel-stable,host-os-linux-64,lib,target-os-linux-64,v18,v18.01,v18.1,v18.1.0,vdefault,vgraph,vneon,vopencl
4cc4967546a67c59   linux-64    64 ARM Compute Library (opencl)  17.12-48bc34ea   64bits,arm,arm-compute-library,armcl,channel-stable,host-os-linux-64,lib,target-os-linux-64,v17,v17.12,v17.12.0,vdefault,vgraph,vneon,vopencl
```

### Install MobileNets weights

To install all the MobileNets-v1 weights in one go:
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

### Evaluate performance
```
$ cd `ck find script:mobilenets-armcl-opencl`
$ python benchmark.py [--repetitions=10]
```

### Evaluate accuracy
```
$ cd `ck find script:mobilenets-armcl-opencl`
$ python benchmark.py --accuracy
```

### Check experimental results
```
$ ck list local:experiment:*
```
 or:
```
$ ck search experiment --tags=request-asplos18
```
 or:
```
$ ck search experiment --tags=request-asplos18,performance
```

If something goes wrong, we suggest you remove experimental results before starting new exploration as follows:

```
$ ck rm experiment:* --tags=request-asplos18 [--force]
```

You can also create a compressed file with experimental results as follows (by default, `ckr-local.zip`):

```
$ ck zip local:experiment:* [--archive_name=<archive name>.zip]
```

You can then copy the resulting file to another machine, and extract it into a CK repository as follows:
```
$ ck unzip repo:<repo name> --zip=<archive name>.zip
```

### Shared experimental results 

We have shared raw experimental results in the CK format 
in this [CK repo](https://github.com/ctuning/ck-request-asplos18-results-mobilenets-armcl-opencl).

You can view and test them as follows:
```
$ ck pull repo:ck-request-asplos18-results-mobilenets-armcl-opencl
$ ck ls ck-request-asplos18-results-mobilenets-armcl-opencl:experiment:* | sort
$ ck dashboard request.apslos18
```

### Unify output and add extra dimensions

Scripts to unify all experiments and add extra dimensions in the ReQuEST format for further comparison and visualization are available here:
```
$ ck find ck-request-asplos18-mobilenets-armcl-opencl:script:mobilenets-armcl-opencl
```

- `benchmark-merge-performance-with-accuracy.py`: merges separately obtained performance and accuracy data together;
- `benchmark-add-dimensions.py`: adds extra dimensions (e.g. cost, peak power consumption).

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


### Make a sample run

```
$ ck run ck-tensorflow:program:classification-tensorflow
...
Model module: /home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-py/mobilenet-model.py
Model weights: /home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-py/mobilenet_v1_1.0_224.ckpt
Input images dir: /home/anton/ilsvrc2012_val
Batch size: 1
Batch count: 1
Net created in 2.963983s
Restore checkpoints from /home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-py/mobilenet_v1_1.0_224.ckpt
Weights loaded in 2.038522s

Batch 0
Batch loaded in 0.293845s
Batch classified in 0.506592s
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.73 - (65) n01751748 sea snake
0.07 - (67) n01755581 diamondback, diamondback rattlesnake, Cr...
0.06 - (53) n01728920 ringneck snake, ring-necked snake, ring ...
0.04 - (60) n01740131 night snake, Hypsiglena torquata
0.03 - (54) n01729322 hognose snake, puff adder, sand viper
---------------------------------------


Average classification time: 0.506592s
Accuracy top 1: 1.000000 (1 of 1)
Accuracy top 5: 1.000000 (1 of 1)


All batches time: 5.811523s
Execution time: 10.881770s


  (post processing from script  /  ... )"


  (reading fine grain timers from tmp-ck-timer.json ...)

{
  "CK_BATCH_COUNT": 1,
  "CK_BATCH_SIZE": 1,
  "CK_IMAGENET_SYNSET_WORDS_TXT": "/home/anton/CK_TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt",
  "CK_IMAGENET_VAL_TXT": "/home/anton/CK_TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt",
  "CK_MODEL_MODULE": "/home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-py/mobilenet-model.py",
  "CK_MODEL_WEIGHTS": "/home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-py/mobilenet_v1_1.0_224.ckpt",
  "accuracy_top1": 1.0,
  "accuracy_top5": 1.0,
  "avg_fps": 1.9739749745976212,
  "avg_time_ms": 506.5920352935791,
  "batch_size": 1,
  "batch_time_ms": 506.5920352935791,
  "execution_time": 10.881769895553589,
  "frame_predictions": [
    {
      "accuracy_top1": "yes",
      "accuracy_top5": "yes",
      "class_correct": 65,
      "class_topmost": 65,
      "file_name": "ILSVRC2012_val_00000001.JPEG"
    }
  ],
  "images_load_time_s": 0.29384493827819824,
  "net_create_time_s": 2.9639828205108643,
  "prediction_time_avg_s": 0.5065920352935791,
  "prediction_time_total_s": 0.5065920352935791,
  "total_time_ms": 5811.522960662842,
  "weights_load_time_s": 2.0385220050811768
}


Execution time: 10.882 sec.
```

### Evaluate performance
```
$ cd `ck find script:mobilenets-tensorflow`
$ python benchmark.py [--repetitions=10]
```

### Evaluate accuracy
```
$ cd `ck find script:mobilenets-tensorflow`
$ python benchmark.py --accuracy
```
