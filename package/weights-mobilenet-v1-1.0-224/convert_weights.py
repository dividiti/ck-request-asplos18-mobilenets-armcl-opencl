#
# Script is based on this gist
# https://gist.github.com/StanislawAntol/656e3afe2d43864bb410d71e1c5789c1#file-freeze_mobilenet-py
# and ARM's conversion script
# https://github.com/ARM-software/ComputeLibrary/blob/master/scripts/tensorflow_data_extractor.py
#
# We can't directly use ARM's conversion script because of the open-source TensorFlow can't
# load metagraphs for models listed in 
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
# See this issue for details: https://github.com/tensorflow/models/issues/1564
# So we need to build a MobileNet model using module mobilenet_v1.py and restore checkpoints into it.
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
#

import os
import shutil
import tensorflow as tf
import numpy as np
import mobilenet_v1

SOURCE_PATH = ''
TARGET_PATH = os.path.join('.', 'npy')
MULTIPLIER = os.getenv('MOBILENET_MULTIPLIER')
RESOLUTION = os.getenv('MOBILENET_RESOLUTION')

if os.path.isdir(TARGET_PATH):
  shutil.rmtree(TARGET_PATH)
os.mkdir(TARGET_PATH)

with tf.Session() as sess:
  input_shape = (None, int(RESOLUTION), int(RESOLUTION), 3)
  input_node = tf.placeholder(tf.float32, shape=input_shape, name="input")
  mobilenet_v1.mobilenet_v1(input_node, 
                            num_classes = 1001, 
                            is_training = False, 
                            depth_multiplier = float(MULTIPLIER))

  saver = tf.train.Saver()
  ckpt_file_prefix = 'mobilenet_v1_{}_{}.ckpt'.format(MULTIPLIER, RESOLUTION)
  saver.restore(sess, os.path.join(SOURCE_PATH, ckpt_file_prefix))

  for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    varname = t.name
    if os.path.sep in t.name:
      varname = varname.replace(os.path.sep, '_')
    if varname.startswith('MobilenetV1_'):
      varname = varname[12:]
    if varname.endswith(':0'):
      varname = varname[:-2]
    target_file = os.path.join(TARGET_PATH, varname)
    print("Saving variable {0} with shape {1} ...".format(varname, t.shape))
    np.save(target_file, sess.run(t))

