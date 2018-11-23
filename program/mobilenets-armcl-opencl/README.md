# MobileNets-v1 program using Arm Compute Library

ImageNet classification and benchmarking using ArmCL and MobileNets-v1.

## Requirements

### ArmCL library
To build this program, you need ArmCL compiled with Graph API:
```bash
$ ck install package:lib-armcl-opencl-18.08 --env.USE_GRAPH=ON --env.USE_NEON=ON --extra_version=-graph
```

To build this program for Android you need to embed OpenCL kernels and select the target API as follows:
```bash
$ ck install package:lib-armcl-opencl-18.08 --env.USE_GRAPH=ON --env.USE_NEON=ON --extra_version=-graph \
--env.USE_EMBEDDED_KERNELS=ON --target_os=android23-arm64 [--env.DEBUG=ON]
```

**NB:** We have to embed kernels when building for Android as OpenCL kernel files are not copied to a remote device.

**NB:** Use `--target_os=android23-arm64` to build for Android API 23 (v6.0 "Marshmallow") or [similar](https://source.android.com/setup/start/build-numbers).

**TODO:** For some reason only a debug build of the library can be used with this program on some Android devices.
(When a release version is used, the program appears to get stuck at the graph preparation stage.)

### Install pretrained and converted weights
Install one or more of the compatible MobileNet weights packages:
```bash
$ ck install package --tags=mobilenet,weights,npy
```

### Install ImageNet validation dataset (50,000 images)
```bash
$ ck install package:imagenet-2012-aux
$ ck install package:imagenet-2012-val
```

## Build
```bash
$ ck compile program:mobilenet-armcl-opencl [--target_os=android23-arm64] 
```

## Run
```bash
$ ck run program:mobilenet-armcl-opencl [--target_os=android23-arm64] 
```

## Benchmark

```bash
$ ck benchmark program:mobilenet-armcl-opencl [--target_os=android23-arm64] [--repetitions=10] [--dvdt_prof]
```

## Program parameters

### ArmCL parameters

Define a parameter by passing `--env.<NAME>=<VALUE>` to `ck run program:mobilenets-armcl-opencl`.
See a [common header](https://github.com/ctuning/ck-math/blob/master/program/armcl-classification-mobilenet/armcl_graph_common.h) for more details.

#### `CK_CONVOLUTION_METHOD`
- `DEFAULT`: use library defaults (possibly a mix of methods).
- `GEMM`: use GEMM-based convolutions.
- `DIRECT`: use direct convolutions.
- `WINOGRAD`: use Winograd convolutions (supported from v18.08; not applicable to MobileNets).

#### `CK_LWS_TUNER_TYPE`

- `NONE`: do not use any tuner (default).
- `DEFAULT`: use the default `CLTuner` (preferred).
- `BIFROST`: use the static `BifrostTuner` (may be deprecated).

#### `CK_DATA_LAYOUT`

- `NCHW` (default).
- `NHWC` (supported from v18.08).
