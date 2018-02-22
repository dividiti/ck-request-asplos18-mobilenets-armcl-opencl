/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <arm_compute/graph/Graph.h>
#include <arm_compute/graph/Nodes.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CPP/CPPScheduler.h>
#include <arm_compute/runtime/Scheduler.h>
#include <support/ToolchainSupport.h>
#include <arm_compute/runtime/CL/CLTuner.h>
#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>

#include "GraphTypePrinter.h"
#include "GraphUtils.h"
#include "Utils.h"

#include <xopenme.h>

#include <cstdlib>
#include <iostream>
#include <string> 
#include <chrono>

enum GLOBAL_TIMER {
  X_TIMER_SETUP,
  X_TIMER_TEST,

  GLOBAL_TIMER_COUNT
};

enum GLOBAL_VAR {
  VAR_TIME_SETUP,
  VAR_TIME_TEST,
  VAR_TIME_IMG_LOAD_TOTAL,
  VAR_TIME_IMG_LOAD_AVG,
  VAR_TIME_CLASSIFY_TOTAL,
  VAR_TIME_CLASSIFY_AVG,

  GLOBAL_VAR_COUNT
};

using namespace std;
using namespace arm_compute;
using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

inline int getenv_i(const char* name, int def) {
    return getenv(name) ? atoi(getenv(name)) : def;
}

inline float getenv_f(const char* name, float def) {
    return getenv(name) ? atof(getenv(name)) : def;
}

inline void store_value_f(int index, const char* name, float value) {
    char* json_name = new char[strlen(name) + 6];
    sprintf(json_name, "\"%s\":%%f", name);
    xopenme_add_var_f(index, json_name, value);
    delete[] json_name;
}

inline int get_batch_size() {
    return getenv_i("CK_BATCH_SIZE", 1);
}

inline int get_batch_count() {
    return getenv_i("CK_BATCH_COUNT", 1);
}

inline const char* get_weights_path() {
    return getenv("CK_ENV_MOBILENET");
}

inline int get_image_size() {
    return getenv_i("CK_ENV_MOBILENET_RESOLUTION", 1);
}

inline const char* get_labels_file() {
	return getenv("CK_CAFFE_IMAGENET_SYNSET_WORDS_TXT");
}

inline const char* get_images_list() {
	return getenv("CK_IMG_LIST");
}

inline const char* get_batches_list() {
	return getenv("CK_BATCH_LIST");
}

inline const char* get_result_dir() {
	return getenv("CK_RESULTS_DIR");
}

inline float get_multiplier() {
  return getenv_f("CK_ENV_MOBILENET_MULTIPLIER", 1);
}

inline ConvolutionMethodHint get_convolution_hint() {
    const arm_compute::GPUTarget gpu_target = arm_compute::CLScheduler::get().target();
    return (gpu_target == arm_compute::GPUTarget::BIFROST ? ConvolutionMethodHint::DIRECT : ConvolutionMethodHint::GEMM);
}

inline char path_separator()
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

inline bool file_exists(const string& name) {
    ifstream f(name);
    return f.good();
}

#ifndef DATATYPE
#define DATATYPE DataType::F32
#endif

class CKPredictionSession {
public:
  const vector<string>& image_files() const { return _image_files; }
  const vector<string>& batch_files() const { return _batch_files; }
  int batch_index() const { return _batch_index; }
  size_t batch_size() const { return _batch_size; }
  size_t batch_count() const { return _batch_files.size(); }
  size_t image_size() const { return _image_size; }
  float total_load_images_time() const { return _total_load_images_time; }
  float total_prediction_time() const { return _total_prediction_time; }
  
  void init() {
    _batch_index = -1;
    _batch_size  = get_batch_size();
    _image_size  = get_image_size();
    _total_load_images_time = 0;
    _total_prediction_time = 0;

    load_file_list();
  }
  
  string get_next_batch_file() {
    if (_batch_index+1 >= _batch_files.size())
      return string();
    _batch_index++;
    return _batch_files[_batch_index];
  }
  
  void measure_begin() {
    _start_time = chrono::high_resolution_clock::now();
  }
  
  float measure_end() {
    auto finish_time = chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish_time - _start_time;
    return elapsed.count();
  }

  float measure_end_load_images() {
    auto duration = measure_end();
    _total_load_images_time += duration;
    return duration;
  }
  
  float measure_end_prediction() {
    auto duration = measure_end();
    _total_prediction_time += duration;
    return duration;
  }
  
private:
  int _batch_index;
  size_t _batch_size;
  size_t _image_size;
  vector<string> _image_files;
  vector<string> _batch_files;
  float _total_load_images_time;
  float _total_prediction_time;
  chrono::time_point<chrono::high_resolution_clock> _start_time;
  
  // TODO: Currently each batch consists of a single image, but it's not general case 
  // and additional work should be done to process real batches
  // https://github.com/ARM-software/ComputeLibrary/issues/355
  void load_file_list() {
    ifstream img_list(get_images_list());
    for (string file_name; !getline(img_list, file_name).fail();)
      _image_files.emplace_back(file_name);

    ifstream batch_list(get_batches_list());
    for (string file_name; !getline(batch_list, file_name).fail();)
      _batch_files.emplace_back(file_name);

    if (_batch_size != 1 || _image_files.size() != _batch_files.size())
       throw runtime_error("Only single image batches are currently supported");
  }
};

inline CKPredictionSession& session() {
  static CKPredictionSession s;
  return s;
}

#endif // BENCHMARK_H
