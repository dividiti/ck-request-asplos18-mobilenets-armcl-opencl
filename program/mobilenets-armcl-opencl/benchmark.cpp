/*
 * Copyright (c) 2018 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include "benchmark.h"

void run_mobilenet();

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
    ofstream err_log("test_errors.log", ios::trunc);
    try
    {
        session().init();

        run_mobilenet();

        return EXIT_SUCCESS;
    }
    catch (cl::Error &err)
    {
        ostringstream msg;
        msg << "\nERROR: " << err.what() << " (" << err.err() << ")";
        cerr << msg.str() << endl;
        err_log << msg.str() << endl;
        return EXIT_FAILURE;
    }
    catch (std::runtime_error &err)
    {
        ostringstream msg;
        msg << "\nERROR: " << err.what() << " " << (errno ? strerror(errno) : "");
        cerr << msg.str() << endl;
        err_log << msg.str() << endl;
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
