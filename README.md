# [dividiti](http://dividiti.com)'s submission to [ReQuEST @ ASPLOS'18](http://cknowledge.org/request-cfp-asplos2018.html)

## Installation

### Install Collective Knowledge

```
# pip install ck
```

### Get submission repository

```
$ ck pull repo --url=https://github.com/dividiti/request-asplos18-mobilenets-armcl-opencl
```

### Pre-install dependencies

**NB:** We suggest to pre-install the following dependencies to test the workflows:

```
$ ck install ck-caffe:package:imagenet-2012-aux
$ ck install ck-caffe:package:imagenet-2012-val-min-resized
$ ck install ck-math:package:lib-armcl-opencl-18.01 --env.USE_GRAPH=ON --env.USE_NEON=ON
$ ck install request-asplos18:package:weights-mobilenet-v1-1.0-224-npy
```

## Build and make a sample run

```
$ ck compile request-asplos18:program:mobilenets-armcl-opencl
$ ck run request-asplos18:program:mobilenets-armcl-opencl
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
