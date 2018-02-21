#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#

import os
import re
import shutil  
import numpy as np
import scipy.io
from scipy.ndimage import zoom

def recreate_dir(d):
  if os.path.isdir(d):
    shutil.rmtree(d)
  os.mkdir(d)

def ck_preprocess(i):

  nenv={} # new environment to be added to the run script

  print('\n--------------------------------')
  def my_env(var): return i['env'][var]
  def dep_env(dep, var): return i['deps'][dep]['dict']['env'][var]

  # Init variables from environment
  BATCH_COUNT = int(my_env('CK_BATCH_COUNT'))
  BATCH_SIZE = int(my_env('CK_BATCH_SIZE'))
  IMAGES_COUNT = BATCH_COUNT * BATCH_SIZE
  SKIP_IMAGES = int(my_env('CK_SKIP_IMAGES'))
  IMAGE_LIST = my_env('CK_IMG_LIST')
  IMAGE_DIR = dep_env('imagenet-val', 'CK_ENV_DATASET_IMAGENET_VAL') 

  S_IMAGE_SIZE=dep_env('weights', 'CK_ENV_MOBILENET_RESOLUTION')
  IMAGE_SIZE = int(S_IMAGE_SIZE)
#  IMAGE_SIZE = int(my_env('CK_ENV_MOBILENET_RESOLUTION')) 
  nenv['CK_ENV_MOBILENET_RESOLUTION']=S_IMAGE_SIZE

  S_MOBILENET_WIDTH_MULTIPLIER = dep_env('weights', 'CK_ENV_MOBILENET_MULTIPLIER')
  MOBILENET_WIDTH_MULTIPLIER = float(S_MOBILENET_WIDTH_MULTIPLIER)
#  MOBILENET_WIDTH_MULTIPLIER = float(my_env('CK_ENV_MOBILENET_WIDTH_MULTIPLIER'))
  nenv['CK_ENV_MOBILENET_WIDTH_MULTIPLIER']=S_MOBILENET_WIDTH_MULTIPLIER

  BATCHES_DIR = my_env('CK_BATCHES_DIR')
  BATCH_LIST = my_env('CK_BATCH_LIST')
  RESULTS_DIR = my_env('CK_RESULTS_DIR')

  def prepare_batches():  
    print('Prepare images...')
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Batch count: {}'.format(BATCH_COUNT))
    print('Batch list: {}'.format(BATCH_LIST))
    print('Skip images: {}'.format(SKIP_IMAGES))
    print('Image dir: {}'.format(IMAGE_DIR))
    print('Image list: {}'.format(IMAGE_LIST))
    print('Image size: {}'.format(IMAGE_SIZE))
    print('Batches dir: {}'.format(BATCHES_DIR))
    print('Results dir: {}'.format(RESULTS_DIR))

    # Load processing image filenames
    images = []
    assert os.path.isdir(IMAGE_DIR), 'Input dir does not exit'
    files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    files = [f for f in files if re.search(r'\.jpg$', f, re.IGNORECASE)
                              or re.search(r'\.jpeg$', f, re.IGNORECASE)]
    assert len(files) > 0, 'Input dir does not contain image files'
    files = sorted(files)[SKIP_IMAGES:]
    assert len(files) > 0, 'Input dir does not contain more files'
    images = files[:IMAGES_COUNT]
    if len(images) < IMAGES_COUNT:
      for _ in range(IMAGES_COUNT-len(images)):
        images.append(images[-1])

    # Save image list file
    assert IMAGE_LIST, 'Image list file name is not set'
    with open(IMAGE_LIST, 'w') as f:
      for img in images:
        f.write('{}\n'.format(img))

    # Preprocess and convert images
    
    dst_images = []

    for img_file in images:
      src_img_path = os.path.join(IMAGE_DIR, img_file)
      dst_img_path = os.path.join(BATCHES_DIR, img_file) + '.npy'

      img = scipy.misc.imread(src_img_path)
      # The same image preprocessing steps are used for MobileNet as for Inseption:
      # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

      # Crop the central region of the image with an area containing 87.5% of the original image.
      new_w = int(img.shape[0] * 0.875)
      new_h = int(img.shape[1] * 0.875)
      offset_w = (img.shape[0] - new_w)/2
      offset_h = (img.shape[1] - new_h)/2
      img = img[offset_w:new_w+offset_w, offset_h:new_h+offset_h, :]

      # Convert to float and normalize
      img = img.astype(np.float32)
      img = img / 255.0

      # Zoom to target size
      zoom_w = float(IMAGE_SIZE)/float(img.shape[0])
      zoom_h = float(IMAGE_SIZE)/float(img.shape[1])
      img = zoom(img, [zoom_w, zoom_h, 1])

      # Shift and scale
      img = img - 0.5
      img = img * 2   
      
      # Each image is a batch in NCHW format
      img = img.transpose(2, 0, 1)
      img = np.expand_dims(img, 0)
      img = np.ascontiguousarray(img)  

      np.save(dst_img_path, img) 
      dst_images.append(dst_img_path)

    # Save image list file
    assert BATCH_LIST, 'Batch list file name is not set'
    with open(BATCH_LIST, 'w') as f:
      for img in dst_images:
        f.write('{}\n'.format(img))


  # Prepare directories
  recreate_dir(BATCHES_DIR)
  recreate_dir(RESULTS_DIR)
  prepare_batches()

  print('--------------------------------\n')
  return {'return': 0, 'new_env':nenv}

