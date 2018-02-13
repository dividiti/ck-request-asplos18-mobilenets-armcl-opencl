# [dividiti](http://dividiti.com)'s submission to [ReQuEST @ ASPLOS'18](http://cknowledge.org/request-cfp-asplos2018.html)

## Artifact check-list (meta-information)

We use standard [Artifact Evaluation check-list](http://ctuning.org/ae/submission_extra.html) from CGO, PPoPP, PACT and SuperComputing and other systems conferences.

* **Algorithm:** image classification
* **Program:** Arm Compute Library v18+ with MobileNets
* **Compilation:** GCC v6+; Python 2.7+ or 3.4+
* **Transformations:** 
* **Binary:** will be compiled on a target platform
* **Data set:** ImageNet
* **Run-time environment:** Linux; OpenCL drivers v1.2+
* **Hardware:** HiKey 960 Development Board (or any similar)
* **Run-time state:** 
* **Execution:** CPU and GPU frequency set to maximum
* **Metrics:** total execution time; accuracy after validating some(all) images from the data set
* **Output:** classification result; execution time; accuracy
* **Experiments:** CK command line 
* **How much disk space required (approximately)?:**: 
* **How much time is needed to prepare workflow (approximately)?:**: 20 minutes
* **How much time is needed to complete experiments (approximately)?**:
* **Collective Knowledge workflow framework used?:** Yes
* **Publicly available?:** Yes

## Installation

### Install global prerequisites

Note that *#* sign means *sudo* on Linux

```
# sudo apt-get install python python-pip sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev python-numpy python-scipy
# sudo pip install pillow
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

### Pre-install CK dependencies

**NB:** We suggest to pre-install the following dependencies to test the workflows:

```
$ ck install ck-caffe:package:imagenet-2012-aux
$ ck install ck-caffe:package:imagenet-2012-val-min-resized
$ ck install ck-math:package:lib-armcl-opencl-18.01 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-1.0-224-npy
```

## Build and make a sample run

```
$ ck compile ck-request-asplos18-mobilenets-armcl-opencl:program:mobilenets-armcl-opencl
$ ck run ck-request-asplos18-mobilenets-armcl-opencl:program:mobilenets-armcl-opencl
...
--------------------------------
Process results in predictions
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.86 - (144) n02051845 pelican
0.03 - (706) n03899768 patio, terrace
0.02 - (969) n07932039 eggnog
0.01 - (369) n02483708 siamang, Hylobates syndactylus, Symphala...
0.01 - (455) n02877765 bottlecap
---------------------------------------
Accuracy top 1: 0.000000 (0 of 1)
Accuracy top 5: 0.000000 (0 of 1)
--------------------------------


  (reading fine grain timers from tmp-ck-timer.json ...)

{
  "accuracy_top1": 0.0,
  "accuracy_top5": 0.0,
  "frame_predictions": [
    {
      "accuracy_top1": "no",
      "accuracy_top5": "no",
      "class_correct": 65,
      "class_topmost": 144,
      "file_name": "ILSVRC2012_val_00000001.JPEG"
    }
  ],
  "images_load_time_avg_s": 0.017375,
  "images_load_time_s": 0.017375,
  "prediction_time_avg_s": 0.058928,
  "prediction_time_total_s": 0.058928,
  "setup_time_s": 1.164188,
  "test_time_s ": 0.083443
}
```
## Exploring performance and accuracy of MobileNet family
Brief description. 
See the the paper: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
### Install MobileNet's weights

To install weights for images with resolution 224x224
```
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-1.0-224-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.75-224-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.50-224-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.25-224-npy
```

To install weights for images with resolution 192x192
```
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-1.0-192-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.75-192-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.50-192-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.25-192-npy
```

To install weights for images with resolution 160x160
```
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-1.0-160-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.75-160-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.50-160-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.25-160-npy
```

To install weights for images with resolution 128x128
```
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-1.0-128-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.75-128-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.50-128-npy
ck install ck-request-asplos18-mobilenets-armcl-opencl:package:weights-mobilenet-v1-0.25-128-npy
```
### Make a simple run
Test a scaled MobileNet architecture, please specify:
 - **resolution** : 224 (default), 192, 160, 128
 - **width_multiplier** : 1.0 (default), 0.75, 0.5, 0.25
```
$ ck run ck-request-asplos18-mobilenets-armcl-opencl:program:mobilenets-armcl-opencl \
      --env.CK_ENV_MOBILENET_RESOLUTION=192   \
      --env.CK_ENV_MOBILENET_WIDTH_MULTIPLIER=0.75 \
```
Then, select the corresponding weight. 

### Performance evaluation of MobileNet family

```
$ python mobilenets-pipe.py --repetitions=3
```

### Accuracy evaluation of MobileNet family

```
$ python mobilenets-pipe.py --accuracy --repetitions=3
```
### Check the experiments

```
$ ck list local:experiment:*
```
