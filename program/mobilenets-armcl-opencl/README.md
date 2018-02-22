# mobilenet-armcl-opencl

ImageNet classification and benchmarking using ArmCL and MobileNet.

## Requirements

ArmCL compiled with Graph API:
```
ck install package:lib-armcl-opencl-18.01 --env.USE_GRAPH=ON --env.USE_NEON=ON --extra_version=-graph
```

One of MobileNet weights packages:
```
ck install package:weights-mobilenet-v1-1.0-224-npy
ck install package:weights-mobilenet-v1-0.75-224-npy
ck install package:weights-mobilenet-v1-0.50-224-npy
ck install package:weights-mobilenet-v1-0.25-224-npy

ck install package:weights-mobilenet-v1-1.0-192-npy
ck install package:weights-mobilenet-v1-0.75-192-npy
ck install package:weights-mobilenet-v1-0.50-192-npy
ck install package:weights-mobilenet-v1-0.25-192-npy

ck install package:weights-mobilenet-v1-1.0-160-npy
ck install package:weights-mobilenet-v1-0.75-160-npy
ck install package:weights-mobilenet-v1-0.50-160-npy
ck install package:weights-mobilenet-v1-0.25-160-npy

ck install package:weights-mobilenet-v1-1.0-128-npy
ck install package:weights-mobilenet-v1-0.75-128-npy
ck install package:weights-mobilenet-v1-0.50-128-npy
ck install package:weights-mobilenet-v1-0.25-128-npy
```

ImageNet dataset:
```
ck install package:imagenet-2012-val
ck install package:imagenet-2012-aux
```

## Build
```
ck compile program:mobilenet-armcl-opencl
```

## Run
```
ck run program:mobilenet-armcl-opencl
```
