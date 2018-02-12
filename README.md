# dividiti's submission to ReQuEST @ ASPLOS'18

## Installation

### Install Collective Knowledge

```
# pip install ck
```

### Get submission repository

```
$ ck pull repo --url=https://github.com/dividiti/request-asplos18
```

### Pre-install dependencies

**NB:** We suggest to pre-install the following dependencies to test the workflows:

```
$ ck install request-asplos18:package:lib-armcl-opencl-18.01 --env.USE_GRAPH=ON --env.USE_NEON=ON --extra_version=-graph
$ ck install request-asplos18:package:weights-mobilenet-v1-1.0-224-npy
$ ck install ck-caffe:package:imagenet-2012-val-min-resized
$ ck install ck-caffe:package:imagenet-2012-aux
```

## Build and run

```
$ ck compile request-asplos18:program:mobilenets-armcl-opencl
$ ck run request-asplos18:program:mobilenets-armcl-opencl
```
