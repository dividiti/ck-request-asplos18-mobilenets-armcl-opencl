/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#pragma once

#if defined(ARMCL_18_05_PLUS)
#include <arm_compute/graph.h>
#include <arm_compute/graph/nodes/Nodes.h>
#include <arm_compute/graph/backends/BackendRegistry.h>
#include <arm_compute/graph/backends/CL/CLDeviceBackend.h>
#else
#include <arm_compute/graph/Graph.h>
#include <arm_compute/graph/Nodes.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#endif

#ifndef DATATYPE
#define DATATYPE DataType::F32
#endif

inline void printf_callback(const char *buffer, unsigned int len, size_t complete, void *user_data) {
  printf("%.*s", len, buffer);
}

inline void set_kernel_path() {
  const char* kernel_path = getenv("CK_ENV_LIB_ARMCL_CL_KERNELS");
  if (kernel_path) {
    printf("Kernel path: %s\n", kernel_path);
    arm_compute::CLKernelLibrary::get().set_kernel_path(kernel_path);
  }
}

inline void init_armcl(arm_compute::ICLTuner *cl_tuner = nullptr) {
  cl_context_properties properties[] =
  {
    CL_PRINTF_CALLBACK_ARM, reinterpret_cast<cl_context_properties>(printf_callback),
    CL_PRINTF_BUFFERSIZE_ARM, static_cast<cl_context_properties>(0x100000),
    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(cl::Platform::get()()),
    0
  };
  cl::Context::setDefault(cl::Context(CL_DEVICE_TYPE_DEFAULT, properties));
  arm_compute::CLScheduler::get().default_init(cl_tuner);

  // Should be called after initialization
  set_kernel_path();

#if defined(ARMCL_18_05_PLUS)
  arm_compute::graph::backends::BackendRegistry::get()
    .add_backend<arm_compute::graph::backends::CLDeviceBackend>(
      arm_compute::graph::Target::CL);
#endif
}

#if defined(ARMCL_18_05_PLUS)

inline arm_compute::graph::ConvolutionMethod str_to_convolution_method(const char *method_name) {
  if (!method_name || strlen(method_name) == 0)
    return arm_compute::graph::ConvolutionMethod::DEFAULT;
    
  // Try to get convolution method by its name
  if (strcmp(method_name, "DEFAULT") == 0) return arm_compute::graph::ConvolutionMethod::DEFAULT;
  if (strcmp(method_name, "GEMM") == 0) return arm_compute::graph::ConvolutionMethod::GEMM;
  if (strcmp(method_name, "DIRECT") == 0) return arm_compute::graph::ConvolutionMethod::DIRECT;
  if (strcmp(method_name, "WINOGRAD") == 0) return arm_compute::graph::ConvolutionMethod::WINOGRAD;
  
  // Try to get convolution method as integer value.
  switch (atoi(method_name)) {
    case 0: return arm_compute::graph::ConvolutionMethod::GEMM;
    case 1: return arm_compute::graph::ConvolutionMethod::DIRECT;
    case 2: return arm_compute::graph::ConvolutionMethod::WINOGRAD;
  }
  
  return arm_compute::graph::ConvolutionMethod::DEFAULT;
}

inline arm_compute::graph::ConvolutionMethod get_convolution_method() {
  auto method_name = getenv("CK_CONVOLUTION_METHOD");
  if (!method_name) {
      bool bifrost_target = (arm_compute::CLScheduler::get().target() == arm_compute::GPUTarget::BIFROST);
      return (bifrost_target ? arm_compute::graph::ConvolutionMethod::DIRECT : arm_compute::graph::ConvolutionMethod::GEMM);
  }
  return str_to_convolution_method(method_name);
}

inline arm_compute::graph::Target get_target_hint() {
  return arm_compute::graph::Target::CL;
}

#define GRAPH(graph_var, graph_name)\
  arm_compute::graph::frontend::Stream graph_var{ 0, graph_name };

#else // ArmCL < 18.05

inline arm_compute::graph::ConvolutionMethodHint str_to_convolution_method(const char *method_name) {
  if (!method_name || strlen(method_name) == 0)
    return arm_compute::graph::ConvolutionMethodHint::GEMM;

  // Try to get convolution method by its name
  if (strcmp(method_name, "GEMM") == 0) return arm_compute::graph::ConvolutionMethodHint::GEMM;
  if (strcmp(method_name, "DIRECT") == 0) return arm_compute::graph::ConvolutionMethodHint::DIRECT;
  
  // Try to get convolution method as integer value.
  switch (atoi(method_name)) {
    case 0: return arm_compute::graph::ConvolutionMethodHint::GEMM;
    case 1: arm_compute::graph::ConvolutionMethodHint::DIRECT;
  }
  
  return arm_compute::graph::ConvolutionMethodHint::GEMM;
}

inline arm_compute::graph::ConvolutionMethodHint get_convolution_method() {
  auto method_name = getenv("CK_CONVOLUTION_METHOD");
  if (method_name)
    return str_to_convolution_method(method_name);

  if (arm_compute::CLScheduler::get().target() == arm_compute::GPUTarget::BIFROST)
    return arm_compute::graph::ConvolutionMethodHint::DIRECT;
        
  return arm_compute::graph::ConvolutionMethodHint::GEMM;
}

inline arm_compute::graph::TargetHint get_target_hint() {
  return arm_compute::graph::TargetHint::OPENCL;
}

#define GRAPH(graph_var, graph_name) \
  arm_compute::graph::Graph graph_var;

#endif // ArmCL < 18.05
