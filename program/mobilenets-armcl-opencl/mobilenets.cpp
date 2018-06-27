/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "benchmark.h"

#include <GraphUtils.h>
#include <utils/Utils.h>

using namespace arm_compute;
using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;
#if defined(ARMCL_18_05_PLUS)
using namespace arm_compute::graph::frontend;
#endif

class CKNumPyInputLoader : public ITensorAccessor {
public:
  CKNumPyInputLoader() {}
  CKNumPyInputLoader(CKNumPyInputLoader &&) = default;

  bool access_tensor(ITensor &tensor) override {
    CKPredictionSession& s = session();
    auto batch_file = s.get_next_batch_file();
    if (batch_file.empty())
      return false;
      
    cout << endl;
    cout << "Batch " << s.batch_index()+1 << " of " << s.batch_count() << endl;
    cout << "File: " << batch_file << endl;
    
    s.measure_begin();

    copy_int8_numpy_to_tensor(batch_file, tensor);

    auto t = s.measure_end_load_images();
    cout << "Loaded in " << t << " s\n";
    
    // Start batch timer after data was loaded
    s.measure_begin();
    return true;
  }

private:
  void copy_int8_numpy_to_tensor(const string& file_name, ITensor &tensor) {
    // Open file
    ifstream stream(file_name, ios::in | ios::binary);
    if (!stream.good())
      raise_error(file_name, "Unable to open file");
    string header = npy::read_header(stream);

    // Parse header
    string typestr;
    bool fortran_order = false;
    vector<unsigned long> shape;
    npy::parse_header(header, typestr, fortran_order, shape);

    // Check if the typestring matches the given one
    if (typestr != arm_compute::utils::get_typestring(DataType::U8))
      raise_error(file_name, "Typestrings mismatch");

    // Reverse vector in case of non fortran order
    if(!fortran_order)
      reverse(shape.begin(), shape.end());

    // Correct dimensions (Needs to match TensorShape dimension corrections)
    const TensorShape tensor_shape = tensor.info()->tensor_shape();
    if(shape.size() != tensor_shape.num_dimensions())
      for(int i = static_cast<int>(shape.size()) - 1; i > 0; --i)
        if(shape[i] == 1)
          shape.pop_back();
        else
          break;

    // Validate tensor ranks and shapes
    if (shape.size() != tensor_shape.num_dimensions())
      raise_error(file_name, "Tensor ranks mismatch");
    for(size_t i = 0; i < shape.size(); ++i)
      if (tensor_shape[i] != shape[i])
        raise_error(file_name, "Tensor dimensions mismatch");

    // Read data
    Window window;
    window.use_tensor_dimensions(tensor_shape);
    execute_window_loop(window, [&](const Coordinates & id) {
      uint8_t value_i8;
      stream.read(reinterpret_cast<char*>(&value_i8), 1);
      float value_f32 = (static_cast<float>(value_i8) / 255.0f - 0.5f) * 2.0f;
      auto target_ptr = reinterpret_cast<float*>(tensor.ptr_to_element(id));
      *target_ptr = value_f32;
    });
  }

  void raise_error(const string& file_name, const string& msg) {
    ostringstream s;
    s << "Failed to read batch file " << file_name << ": " << msg;
    throw runtime_error(s.str());
  }
};


class CKOutputAccessor : public ITensorAccessor {
public:
  CKOutputAccessor() {}
  CKOutputAccessor(CKOutputAccessor &&) = default;

  bool access_tensor(ITensor &tensor) override {
    // Stop batch timer before processing results
    CKPredictionSession& s = session();
    auto t = s.measure_end_prediction();
    cout << "Classified in " << t << "s \n";

    // TODO: some additional work will be required when batch_size > 1 is allowed.
    // We will have to split batch result into a set of results for different images.
    string img_file = s.image_files()[s.batch_index()];
    string res_dir = get_result_dir();
    string res_file = res_dir + path_separator() + img_file + ".txt";
    ofstream f(res_file);

    const size_t num_classes = tensor.info()->dimension(0);
    float* probes  = reinterpret_cast<float*>(tensor.buffer() + tensor.info()->offset_first_element_in_bytes());
    
    // Take off the first probe as it references to 'background' class but no such one in ImageNet
    for (size_t i = 1; i < num_classes; i++)
      f << probes[i] << endl;
      
    return true;
  }
};

std::string get_convolution_methods_file() {
  auto filename = getenv("CK_CONVOLUTION_METHOD_FILE");
  return filename ? std::string(filename) : std::string("conv_methods.txt");
}

std::vector<int> load_convolution_methods() {
  const int convolutions_count = 15;
  const int defaut_method = static_cast<int>(get_convolution_method());

  std::vector<int> methods;
  methods.push_back(defaut_method);
  for (int i = 1; i < convolutions_count-1; i++)
#if defined(ARMCL_18_05_PLUS)
    methods.push_back(static_cast<int>(DepthwiseConvolutionMethod::OPTIMIZED_3x3));
#else
    methods.push_back(static_cast<int>(defaut_method));
#endif
  methods.push_back(defaut_method);

  auto filename = get_convolution_methods_file();
  std::ifstream file(filename);
  if (file) {
    std::cout << "Loading convolutions methods from " << filename << std::endl;
    int index = 0;
    std::string line;
    while (std::getline(file, line) && index < convolutions_count) {
      if (!line.empty()) {
        auto s = line.c_str();
#if defined(ARMCL_18_05_PLUS)
        if (index > 0 || index < convolutions_count-1)
          methods[index] = static_cast<int>(str_to_dwsc_convolution_method(s));
        else
#endif
          methods[index] = static_cast<int>(str_to_convolution_method(s));
        std::cout << "    " << index << ": " << line << std::endl;
      }
      index++;
    }
  }
  return methods;
}

namespace
{
inline unique_ptr<ITensorAccessor> weights_accessor(const string &file)
{
    const string path = get_weights_path();
    string full_path = path + path_separator() + file;
    if (!file_exists(full_path))
    {
       cerr << "WARNING: file not found: " << full_path << ", dummy accessor will be used!\n";
       return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    return arm_compute::support::cpp14::make_unique<NumPyBinLoader>(full_path);
}

inline unique_ptr<ITensorAccessor> empty_accessor() {
    return std::unique_ptr<ITensorAccessor>(nullptr);
}

unsigned int apply_multiplier(unsigned int size) {
    return static_cast<unsigned int>(size * get_multiplier());
}

} // namespace

void run_mobilenet()
{
    auto target_hint        = get_target_hint();
    

    TensorShape input_shape(session().image_size(),
                            session().image_size(),
                            3U,
                            session().batch_size());

    GRAPH(graph, "MobileNetV1");

    auto get_dwsc_node = [&](std::string &&param_path,
                             unsigned int  conv_filt,
                             PadStrideInfo dwc_pad_stride_info,
                             PadStrideInfo conv_pad_stride_info) {
#if defined(ARMCL_18_05_PLUS)
      SubStream sg(graph);
#else
      SubGraph sg;
#endif   
      sg << DepthwiseConvolutionLayer(
             3U, 3U,
             weights_accessor(param_path + "_depthwise_depthwise_weights.npy"),
             empty_accessor(),
             dwc_pad_stride_info)
         << BatchNormalizationLayer(
             weights_accessor(param_path + "_depthwise_BatchNorm_moving_mean.npy"),
             weights_accessor(param_path + "_depthwise_BatchNorm_moving_variance.npy"),
             weights_accessor(param_path + "_depthwise_BatchNorm_gamma.npy"),
             weights_accessor(param_path + "_depthwise_BatchNorm_beta.npy"),
             0.001f)
         << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
         << ConvolutionLayer(
             1U, 1U, apply_multiplier(conv_filt),
             weights_accessor(param_path + "_pointwise_weights.npy"),
             empty_accessor(),
             conv_pad_stride_info)
         << BatchNormalizationLayer(
             weights_accessor(param_path + "_pointwise_BatchNorm_moving_mean.npy"),
             weights_accessor(param_path + "_pointwise_BatchNorm_moving_variance.npy"),
             weights_accessor(param_path + "_pointwise_BatchNorm_gamma.npy"),
             weights_accessor(param_path + "_pointwise_BatchNorm_beta.npy"),
             0.001f)
         << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));
      return BranchLayer(std::move(sg));
    };


    std::vector<int> convolution_methods = load_convolution_methods();

    auto convolution_method = [&](int index){
#if defined(ARMCL_18_05_PLUS)
      return static_cast<arm_compute::graph::ConvolutionMethod>(convolution_methods[index]);
#else
      return static_cast<arm_compute::graph::ConvolutionMethodHint>(convolution_methods[index]);
#endif
    };
    
    auto dwsc_convolution_method = [&](int index){
#if defined(ARMCL_18_05_PLUS)
      return static_cast<arm_compute::graph::DepthwiseConvolutionMethod>(convolution_methods[index]);
#else
      return static_cast<arm_compute::graph::ConvolutionMethodHint>(convolution_methods[index]);
#endif
    };

    std::cout << "\nPrepare graph...\n";
    xopenme_clock_start(X_TIMER_SETUP);
    graph << target_hint
#if defined(ARMCL_18_05_PLUS)
          << InputLayer(TensorDescriptor(input_shape, DATATYPE),
                arm_compute::support::cpp14::make_unique<CKNumPyInputLoader>())
#else
          << arm_compute::graph::Tensor(TensorInfo(input_shape, 1, DATATYPE), 
                arm_compute::support::cpp14::make_unique<CKNumPyInputLoader>())
#endif
          << convolution_method(0) << ConvolutionLayer(
              3U, 3U, apply_multiplier(32U),
              weights_accessor("Conv2d_0_weights.npy"),
              empty_accessor(),
              PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
          << BatchNormalizationLayer(
              weights_accessor("Conv2d_0_BatchNorm_moving_mean.npy"),
              weights_accessor("Conv2d_0_BatchNorm_moving_variance.npy"),
              weights_accessor("Conv2d_0_BatchNorm_gamma.npy"),
              weights_accessor("Conv2d_0_BatchNorm_beta.npy"),
              0.001f)

          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
          << dwsc_convolution_method(1) << get_dwsc_node("Conv2d_1", 64, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(2) << get_dwsc_node("Conv2d_2", 128, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(3) << get_dwsc_node("Conv2d_3", 128, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(4) << get_dwsc_node("Conv2d_4", 256, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(5) << get_dwsc_node("Conv2d_5", 256, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(6) << get_dwsc_node("Conv2d_6", 512, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(7) << get_dwsc_node("Conv2d_7", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(8) << get_dwsc_node("Conv2d_8", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(9) << get_dwsc_node("Conv2d_9", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(10) << get_dwsc_node("Conv2d_10", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(11) << get_dwsc_node("Conv2d_11", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(12) << get_dwsc_node("Conv2d_12", 1024, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << dwsc_convolution_method(13) << get_dwsc_node("Conv2d_13", 1024, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
          << convolution_method(14) << ConvolutionLayer(
              1U, 1U, 1001U,
              weights_accessor("Logits_Conv2d_1c_1x1_weights.npy"),
              weights_accessor("Logits_Conv2d_1c_1x1_biases.npy"),
              PadStrideInfo(1, 1, 0, 0))
          << ReshapeLayer(TensorShape(1001U))
          << SoftmaxLayer()
#if defined(ARMCL_18_05_PLUS)
          << OutputLayer(arm_compute::support::cpp14::make_unique<CKOutputAccessor>());
#else
          << arm_compute::graph::Tensor(arm_compute::support::cpp14::make_unique<CKOutputAccessor>());
#endif
    xopenme_clock_end(X_TIMER_SETUP);

#if defined(ARMCL_18_05_PLUS)
    // Finalize graph
    GraphConfig config {};
    graph.finalize(target_hint, config);
#endif

    std::cout << "\nRun graph...\n";
    xopenme_clock_start(X_TIMER_TEST);
    graph.run();
    xopenme_clock_end(X_TIMER_TEST);
}
