/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include "benchmark.h"

void run_mobilenet();

void printf_callback(const char *buffer, unsigned int len, size_t complete, void *user_data) {
    printf("%.*s", len, buffer);
}

void set_kernel_path() {
     const char* kernel_path = getenv("CK_ENV_LIB_ARMCL_CL_KERNELS");
     if (kernel_path) {
         printf("Kernel path: %s\n", kernel_path);
         arm_compute::CLKernelLibrary::get().set_kernel_path(kernel_path);
     }
}

void init_armcl(arm_compute::ICLTuner *cl_tuner = nullptr) {
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
}

void finish_test() {
    int batch_count = session().batch_count();
    float total_load_images_time = session().total_load_images_time();
    float total_prediction_time = session().total_prediction_time();
    float avg_load_images_time = total_load_images_time / float(batch_count);
    float avg_prediction_time = total_prediction_time / float(batch_count);
    float setup_time = xopenme_get_timer(X_TIMER_SETUP);
    float test_time = xopenme_get_timer(X_TIMER_TEST);
    
    cout << "-------------------------------\n";
    cout << "Graph loaded in " << setup_time << " s" << endl;
    cout << "All batches loaded in " << total_load_images_time << " s" << endl;
    cout << "All batches classified in " << total_prediction_time << " s" << endl;
    cout << "Average classification time: " << avg_prediction_time << " s" << endl;
    cout << "-------------------------------\n";
    
    store_value_f(VAR_TIME_SETUP, "setup_time_s", setup_time);
    store_value_f(VAR_TIME_TEST, "test_time_s ", test_time);
    store_value_f(VAR_TIME_IMG_LOAD_TOTAL, "images_load_time_s", total_load_images_time);
    store_value_f(VAR_TIME_IMG_LOAD_AVG, "images_load_time_avg_s", avg_load_images_time);
    store_value_f(VAR_TIME_CLASSIFY_TOTAL, "prediction_time_total_s", total_prediction_time);
    store_value_f(VAR_TIME_CLASSIFY_AVG, "prediction_time_avg_s", avg_prediction_time);
    
    xopenme_dump_state();
    xopenme_finish();
}

int run_test() {
    try
    {
        session().init();

        run_mobilenet();

        return EXIT_SUCCESS;
    }
    catch (cl::Error &err)
    {
        std::cerr << "\nERROR: " << err.what() << " (" << err.err() << ")" << std::endl;
        return EXIT_FAILURE;
    }
    catch (std::runtime_error &err)
    {
        std::cerr << "\nERROR: " << err.what() << " " << (errno ? strerror(errno) : "") << std::endl;
        return EXIT_FAILURE;
    }
}

int main(int argc, const char **argv) {
    xopenme_init(GLOBAL_TIMER_COUNT, GLOBAL_VAR_COUNT);
    init_armcl();

    int status = run_test();

    if (status == EXIT_SUCCESS)
        std::cout << "Test passed\n";
    else
        std::cout << "Test failed\n";

    finish_test();
    fflush(stdout);
    fflush(stderr);

    return status;
}
