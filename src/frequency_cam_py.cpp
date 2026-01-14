// -*-c++-*--------------------------------------------------------------------
// Copyright 2022 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FREQUENCY_CAM__FREQUENCY_CAM_PY_H_
#define FREQUENCY_CAM__FREQUENCY_CAM_PY_H_

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "frequency_cam/frequency_cam.h"

namespace nb = nanobind;

NB_MODULE(freqcam_ext, m) {
    nb::class_<frequency_cam::FrequencyCam>(m, "FrequencyCam")
        .def(nb::init<>())
        .def("initialize", &frequency_cam::FrequencyCam::initialize)
        .def("initializeState", &frequency_cam::FrequencyCam::initializeState)
        .def("getStatistics", &frequency_cam::FrequencyCam::getStatistics)
        .def("resetStatistics", &frequency_cam::FrequencyCam::resetStatistics)
        .def("makeFrequencyAndEventImage", &frequency_cam::FrequencyCam::makeFrequencyAndEventImage)
        .def("process_batch", [](frequency_cam::FrequencyCam &self, 
                                 nb::ndarray<uint64_t, nb::ndim<1>> ts, 
                                 nb::ndarray<uint16_t, nb::ndim<1>> xs, 
                                 nb::ndarray<uint16_t, nb::ndim<1>> ys, 
                                 nb::ndarray<uint8_t, nb::ndim<1>> pols) {
            for (size_t i = 0; i < ts.shape(0); ++i) {
                self.eventCD(ts(i), xs(i), ys(i), pols(i));
            }
        })
        .def("makeFrequencyAndEventImage", [](frequency_cam::FrequencyCam &self, float dt, bool use_log) {
            // 1. FrequencyCam requires an eventImage pointer to fill, 
            // even if we don't plan to use the overlay.
            cv::Mat eventImg;
    
            // 2. Call the actual C++ function from your header
            // Params: eventImage, overlayEvents, useLogFrequency, dt
            cv::Mat freqImg = self.makeFrequencyAndEventImage(&eventImg, false, use_log, dt);
    
            // 3. Convert the cv::Mat (Float32) to a NumPy ndarray for Python
            size_t shape[2] = {(size_t)freqImg.rows, (size_t)freqImg.cols};
            return nb::ndarray<nb::numpy, float>(freqImg.data, 2, shape, nb::handle());
        });
}

#endif