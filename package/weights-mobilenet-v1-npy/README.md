# MobileNetV1 aggregate package

This package is for installing (or reinstalling) all of its dependencies (all the MobileNetV1 packages) in one go.

First, remove all previously installed weights:
```
$ ck clean env --tags=weights,npy,mobilenet-v1 -f
```

To install all the weights, run:
```
$ ck install package:weights-mobilenet-v1-npy
```
or:
```
$ ck install --tags=mobilenet-v1-all
```
