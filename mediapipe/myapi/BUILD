# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_framework",
)

licenses(["notice"])

###################################################
#Android
###################################################
cc_binary(
    name = "libmediaflow.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":FaceMeshGpuLibraryAndroid"
    ],
    linkopts = [
        "-Wl,-soname,libmediaflow.so",
    ],
)

cc_library(
    name = "FaceMeshGpuLibraryAndroid",
        srcs = [
        "SXFaceLandmarkGPU.cpp",
    ],
    hdrs = [
        "SXMediapipeCommon.h",
        "SXFaceLandmarkGPU.h",
    ],
    copts = ["-std=c++17", "-fvisibility=hidden"],
    alwayslink = 1,
    deps = [
        "//mediapipe/graphs/face_mesh:mobile_calculators",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

###################################################
#IOS
###################################################
MIN_IOS_VERSION = "10.0"

apple_binary(
    name = "libmediaflow.dylib",
    binary_type = "dylib",
    minimum_os_version = MIN_IOS_VERSION,
    platform_type = "ios",
    deps = [
        ":FaceMeshGpuLibraryIOS",
    ],
)

ios_framework(
    name = "Mediaflow",
    bundle_id = "com.sx.mediaflow",
    families = ["iphone"],
    minimum_os_version = MIN_IOS_VERSION,
    infoplists = [":Info.plist"],
    deps = [
        ":FaceMeshGpuLibraryIOS",
    ],
)

objc_library(
    name = "FaceMeshGpuLibraryIOS",
    srcs = [
        "SXFaceLandmarkGPU.mm",
    ],
    hdrs = [
        "SXMediapipeCommon.h",
        "SXFaceLandmarkGPU.h",
    ],
    copts = ["-std=c++17", "-fvisibility=hidden"],
    alwayslink = 1,
    deps = [
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/graphs/face_mesh:mobile_calculators",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gl_calculator_helper_ios",
        "//mediapipe/gpu:gl_quad_renderer",
        "//mediapipe/framework/port:parse_text_proto",
        "@ios_opencv//:OpencvFramework",
    ],
)