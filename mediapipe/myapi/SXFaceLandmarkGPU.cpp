//
// Created by carrot hu on 2021/3/12.
//
#include "SXFaceLandmarkGPU.h"

#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#ifdef __ANDROID__
#include "mediapipe/util/android/asset_manager_util.h"
#endif

#define SIDE_PACKET_NAME_FACE "num_faces"
#define SIDE_PACKET_NAME_GPU_ORIGIN "gpu_origin"
#define INPUT_STREAM "input_video"
#define OUTPUT_LANDMARK_PRESENCE "landmark_presence"
#define OUTPUT_DETECTION_PRESENCE "detection_presence"
#define OUTPUT_LANDMARKS "multi_face_landmarks"
#define OUTPUT_LANDMARK_RECTS "face_rects_from_landmarks"
#define OUTPUT_FACE_DETECTIONS "face_detections"
#define OUTPUT_FACE_DETECTION_RECTS "face_rects_from_detections"

#define LockGuard(m) std::lock_guard<std::mutex> lockGuard(m)

namespace mediapipe
{
struct SXFaceLandmarkGPUGraph
{
    mediapipe::CalculatorGraph mGraph;
    mediapipe::StatusOrPoller mLandmarkPresencePoller;
    mediapipe::StatusOrPoller mDetectionPresencePoller;
    mediapipe::StatusOrPoller mLandmarksPoller;
    mediapipe::StatusOrPoller mLandmarkRectsPoller;
    mediapipe::StatusOrPoller mDetectionsPoller;
    mediapipe::StatusOrPoller mDetectionRectsPoller;

    mediapipe::GlCalculatorHelper mGpuHelper;
    mediapipe::PlatformGlContext mGLContext = nullptr;

    std::mutex mLock;
    std::vector<SXFaceLandmarksData> mLandmarks;
    std::vector<SXFaceDetectionData> mFaceDetections;

    int mMaxFaces;
    bool mBottomLeft;

    SXFaceLandmarkGPUGraph(bool bottomLeft, int maxFaces) : mMaxFaces(maxFaces), mBottomLeft(bottomLeft)
    {
        auto status = create();
        if (!status.ok())
        {
            LOG(ERROR) << status.message();
        }
    };

    ~SXFaceLandmarkGPUGraph()
    {
        stopUtilDone();
    };

    absl::Status create()
    {
        std::string config_contents = "# MediaPipe graph that performs face mesh with TensorFlow Lite on GPU.\n"
                                      "\n"
                                      "# GPU buffer. (GpuBuffer)\n"
                                      "input_stream: \"input_video\"\n"
                                      "\n"
                                      "# Max number of faces to detect/process. (int)\n"
                                      "input_side_packet: \"num_faces\"\n"
                                      "input_side_packet: \"gpu_origin\"  \n"
                                      "\n"
                                      "# Collection of detected/processed faces, each represented as a list of\n"
                                      "# landmarks. (std::vector<NormalizedLandmarkList>)\n"
                                      "output_stream: \"multi_face_landmarks\"\n"
                                      "\n"
                                      "# Extra outputs (for debugging, for instance).\n"
                                      "# Detected faces. (std::vector<Detection>)\n"
                                      "output_stream: \"face_detections\"\n"
                                      "# Regions of interest calculated based on landmarks.\n"
                                      "# (std::vector<NormalizedRect>)\n"
                                      "output_stream: \"face_rects_from_landmarks\"\n"
                                      "# Regions of interest calculated based on face detections.\n"
                                      "# (std::vector<NormalizedRect>)\n"
                                      "output_stream: \"face_rects_from_detections\"\n"
                                      "\n"
                                      "output_stream: \"landmark_presence\"\n"
                                      "\n"
                                      "# Throttles the images flowing downstream for flow control. It passes through\n"
                                      "# the very first incoming image unaltered, and waits for downstream nodes\n"
                                      "# (calculators and subgraphs) in the graph to finish their tasks before it\n"
                                      "# passes through another image. All images that come in while waiting are\n"
                                      "# dropped, limiting the number of in-flight images in most part of the graph "
                                      "to\n"
                                      "# 1. This prevents the downstream nodes from queuing up incoming images and "
                                      "data\n"
                                      "# excessively, which leads to increased latency and memory usage, unwanted in\n"
                                      "# real-time mobile applications. It also eliminates unnecessarily computation,\n"
                                      "# e.g., the output produced by a node may get dropped downstream if the\n"
                                      "# subsequent nodes are still busy processing previous inputs.\n"
                                      "node {\n"
                                      "  calculator: \"FlowLimiterCalculator\"\n"
                                      "  input_stream: \"input_video\"\n"
                                      "  input_stream: \"FINISHED:landmark_presence\"\n"
                                      "  input_stream_info: {\n"
                                      "    tag_index: \"FINISHED\"\n"
                                      "    back_edge: true\n"
                                      "  }\n"
                                      "  output_stream: \"throttled_input_video\"\n"
                                      "}\n"
                                      "\n"
                                      "# Subgraph that detects faces and corresponding landmarks.\n"
                                      "node {\n"
                                      "  calculator: \"FaceLandmarkFrontGpu\"\n"
                                      "  input_stream: \"IMAGE:throttled_input_video\"\n"
                                      "  input_side_packet: \"NUM_FACES:num_faces\"\n"
                                      "  input_side_packet: \"gpu_origin\"  \n"
                                      "  output_stream: \"LANDMARKS:multi_face_landmarks\"\n"
                                      "  output_stream: \"ROIS_FROM_LANDMARKS:face_rects_from_landmarks\"\n"
                                      "  output_stream: \"DETECTIONS:face_detections\"\n"
                                      "  output_stream: \"ROIS_FROM_DETECTIONS:face_rects_from_detections\"\n"
                                      "}\n"
                                      "\n"
                                      "# # Calculate size of the image.\n"
                                      "# node {\n"
                                      "#   calculator: \"ImagePropertiesCalculator\"\n"
                                      "#   input_stream: \"IMAGE_GPU:throttled_input_video\"\n"
                                      "#   output_stream: \"SIZE:image_size\"\n"
                                      "# }\n"
                                      "\n"
                                      "# # Extracts a single set of face landmarks associated with the most prominent\n"
                                      "# # face detected from a collection.\n"
                                      "# node {\n"
                                      "#   calculator: \"SplitNormalizedLandmarkListVectorCalculator\"\n"
                                      "#   input_stream: \"multi_face_landmarks\"\n"
                                      "#   output_stream: \"face_landmarks\"\n"
                                      "#   node_options: {\n"
                                      "#     [type.googleapis.com/mediapipe.SplitVectorCalculatorOptions] {\n"
                                      "#       ranges: { begin: 0 end: 1 }\n"
                                      "#       element_only: true\n"
                                      "#     }\n"
                                      "#   }\n"
                                      "# }\n"
                                      "\n"
                                      "# # Applies smoothing to a face landmark list. The filter options were "
                                      "handpicked\n"
                                      "# # to achieve better visual results.\n"
                                      "# node {\n"
                                      "#   calculator: \"LandmarksSmoothingCalculator\"\n"
                                      "#   input_stream: \"NORM_LANDMARKS:face_landmarks\"\n"
                                      "#   input_stream: \"IMAGE_SIZE:image_size\"\n"
                                      "#   output_stream: \"NORM_FILTERED_LANDMARKS:filtered_landmarks\"\n"
                                      "#   node_options: {\n"
                                      "#     [type.googleapis.com/mediapipe.LandmarksSmoothingCalculatorOptions] {\n"
                                      "#       velocity_filter: {\n"
                                      "#         window_size: 5\n"
                                      "#         velocity_scale: 20.0\n"
                                      "#       }\n"
                                      "#     }\n"
                                      "#   }\n"
                                      "# }\n"
                                      "\n"
                                      "# add PacketPresenceCalculator\n"
                                      "node {\n"
                                      "  calculator: \"PacketPresenceCalculator\"\n"
                                      "  input_stream: \"PACKET:multi_face_landmarks\"\n"
                                      "  output_stream: \"PRESENCE:landmark_presence\"\n"
                                      "}";
        mediapipe::CalculatorGraphConfig config =
            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(config_contents);
        MP_RETURN_IF_ERROR(mGraph.Initialize(config));

        mLandmarkPresencePoller = mGraph.AddOutputStreamPoller(OUTPUT_LANDMARK_PRESENCE);
        if (!mLandmarkPresencePoller.ok())
        {
            return mLandmarkPresencePoller.status();
        }

        // mDetectionPresencePoller =
        // mGraph.AddOutputStreamPoller(OUTPUT_DETECTION_PRESENCE); if
        // (!mDetectionPresencePoller.ok()) {
        //     return mDetectionPresencePoller.status();
        // }

        mLandmarksPoller = mGraph.AddOutputStreamPoller(OUTPUT_LANDMARKS);
        if (!mLandmarksPoller.ok())
        {
            return mLandmarksPoller.status();
        }

        mLandmarkRectsPoller = mGraph.AddOutputStreamPoller(OUTPUT_LANDMARK_RECTS);
        if (!mLandmarkRectsPoller.ok())
        {
            return mLandmarkRectsPoller.status();
        }

        // mDetectionsPoller = mGraph.AddOutputStreamPoller(OUTPUT_FACE_DETECTIONS);
        // if (!mDetectionsPoller.ok()) {
        //     return mDetectionsPoller.status();
        // }

        // mDetectionRectsPoller =
        // mGraph.AddOutputStreamPoller(OUTPUT_FACE_DETECTION_RECTS); if
        // (!mDetectionRectsPoller.ok()) {
        //     return mDetectionRectsPoller.status();
        // }

        return absl::OkStatus();
    }

    absl::Status startRun(void* sharedContext)
    {
        LOG(INFO) << "Initialize the GPU.";
        mGLContext = (mediapipe::PlatformGlContext)sharedContext;
        ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create(mGLContext));
        MP_RETURN_IF_ERROR(mGraph.SetGpuResources(std::move(gpu_resources)));
        MP_RETURN_IF_ERROR(
            mGraph.StartRun({{SIDE_PACKET_NAME_FACE, mediapipe::MakePacket<int>(mMaxFaces)},
                             {SIDE_PACKET_NAME_GPU_ORIGIN, mediapipe::MakePacket<int>(mBottomLeft ? 2 : 1)}}));
        return absl::OkStatus();
    }

    absl::Status process(unsigned texture, int width, int height, SXColorFormat format)
    {
        mediapipe::GlTextureBufferSharedPtr textureBufferSharedPtr = std::make_shared<mediapipe::GlTextureBuffer>(
            GL_TEXTURE_2D, texture, width, height, mediapipe::GpuBufferFormat::kBGRA32,
            [=](std::shared_ptr<mediapipe::GlSyncPoint> sync_token) {});

        auto textureBuffer = absl::make_unique<mediapipe::GpuBuffer>(textureBufferSharedPtr);
        size_t frame_timestamp_us = std::chrono::steady_clock::now().time_since_epoch().count();
        MP_RETURN_IF_ERROR(mGraph.AddPacketToInputStream(
            INPUT_STREAM, mediapipe::Adopt(textureBuffer.release()).At(Timestamp(frame_timestamp_us))));

        mediapipe::Packet presence_packet;
        // mediapipe::Packet detection_presence_packet;
        mediapipe::Packet landmark_packet;
        mediapipe::Packet landmark_rect_packet;
        // mediapipe::Packet detection_packet;
        // mediapipe::Packet detection_rect_packet;

        // if (!mDetectionPresencePoller.value().Next(&detection_presence_packet))
        // return absl::NotFoundError("read packet failed"); auto
        // is_detection_present = detection_presence_packet.Get<bool>();

        if (!mLandmarkPresencePoller.value().Next(&presence_packet))
            return absl::NotFoundError("read packet failed");
        auto is_landmark_present = presence_packet.Get<bool>();

        // if (is_detection_present) {
        //     LockGuard(mLock);
        //     mFaceDetections.clear();
        //     if (mDetectionsPoller.value().Next(&detection_packet)) {
        //         auto detections =
        //         detection_packet.Get<std::vector<mediapipe::Detection>>(); for
        //         (size_t i = 0; i < detections.size(); i++) {
        //             mFaceDetections.emplace_back();
        //             mFaceDetections.back().mScore = detections[i].score().Get(0);
        //         }
        //     }

        //     if (mDetectionRectsPoller.value().Next(&detection_rect_packet)) {
        //         auto rects =
        //         detection_rect_packet.Get<std::vector<mediapipe::NormalizedRect>>();
        //         for (int i = 0; i < rects.size(); ++i) {
        //             mFaceDetections[i].mRect.setSize({rects[i].width(),
        //             rects[i].height()});
        //             mFaceDetections[i].mRect.setCenter({rects[i].x_center(),
        //             rects[i].y_center()}); mFaceDetections[i].mRotation =
        //             rects[i].rotation();
        //         }
        //     }
        // } else {
        //     LockGuard(mLock);
        //     mFaceDetections.clear();
        // }

        if (is_landmark_present)
        {
            LockGuard(mLock);
            mLandmarks.clear();
            // fetch landmarks only when they exist
            if (mLandmarksPoller.value().Next(&landmark_packet))
            {
                auto landmarks = landmark_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                for (auto& item : landmarks)
                {
                    mLandmarks.emplace_back();
                    for (int i = 0; i < item.landmark_size(); ++i)
                    {
                        const auto& landmark = item.landmark(i);
                        mLandmarks.back().mLandmarks.push_back({landmark.x(), landmark.y(), landmark.z()});
                    }
                }
            }

            if (mLandmarkRectsPoller.value().Next(&landmark_rect_packet))
            {
                auto rects = landmark_rect_packet.Get<std::vector<mediapipe::NormalizedRect>>();
                for (int i = 0; i < rects.size(); ++i)
                {
                    if (i < mLandmarks.size())
                    {
                        mLandmarks[i].mRect.push_back(rects[i].x_center() - rects[i].width() * 0.5);
                        mLandmarks[i].mRect.push_back(rects[i].y_center() - rects[i].height() * 0.5);
                        mLandmarks[i].mRect.push_back(rects[i].x_center() + rects[i].width() * 0.5);
                        mLandmarks[i].mRect.push_back(rects[i].y_center() + rects[i].height() * 0.5);
                        mLandmarks[i].mRotation = rects[i].rotation();
                    }
                }
            }
        }
        else
        {
            LockGuard(mLock);
            mLandmarks.clear();
        }

        return absl::OkStatus();
    }

    const std::vector<SXFaceLandmarksData>& getFaceLandmarkData()
    {
        LockGuard(mLock);
        return mLandmarks;
    }

    const SXFaceLandmarksData& getFaceLandmarkData(int index)
    {
        LockGuard(mLock);
        assert(index < mLandmarks.size);
        return mLandmarks[index];
    }

    int getFaceSize()
    {
        LockGuard(mLock);
        return mLandmarks.size();
    }

    const std::vector<SXFaceDetectionData>& getFaceDetectionData()
    {
        LockGuard(mLock);
        return mFaceDetections;
    }

    absl::Status stopUtilDone()
    {
        MP_RETURN_IF_ERROR(mGraph.CloseAllInputStreams());
        MP_RETURN_IF_ERROR(mGraph.CloseAllPacketSources());
        return mGraph.WaitUntilDone();
    }
};
}  // namespace mediapipe

extern "C"
{
    void* sx_createFaceLandmarkGpuGraph(bool bottomLeft, int maxFaces)
    {
        return new mediapipe::SXFaceLandmarkGPUGraph(bottomLeft, maxFaces);
    }

    void sx_destroyFaceLandmarkGpuGraph(void* graph)
    {
        delete (mediapipe::SXFaceLandmarkGPUGraph*)graph;
    }

    bool sx_startFaceGraph(void* graph, void* sharedContext)
    {
        auto status = ((mediapipe::SXFaceLandmarkGPUGraph*)graph)->startRun(sharedContext);
        if (!status.ok())
        {
            LOG(ERROR) << status.message();
        }
        return status.ok();
    }

    bool sx_processPixelbuffer(void* graph, void* pixelbuffer, int width, int height)
    {
        LOG(ERROR) << "Unsupport for current platform";
        return false;
    }

    bool sx_processTexture(void* graph, unsigned texture, int width, int height)
    {
        auto status = ((mediapipe::SXFaceLandmarkGPUGraph*)graph)
                          ->process(texture, width, height, mediapipe::SXColorFormat::kRGBA);
        if (!status.ok())
        {
            LOG(ERROR) << status.message();
        }
        return status.ok();
    }

    bool stopFaceGraph(void* graph)
    {
        auto status = ((mediapipe::SXFaceLandmarkGPUGraph*)graph)->stopUtilDone();
        if (!status.ok())
        {
            LOG(ERROR) << status.message();
        }
        return status.ok();
    }

    int sx_getFaceNum(void* graph)
    {
        return ((mediapipe::SXFaceLandmarkGPUGraph*)graph)->getFaceSize();
    }

    float sx_getFaceLandmarkData(void* graph, int index, float** data, int* data_size, float** rect)
    {
        auto datas = ((mediapipe::SXFaceLandmarkGPUGraph*)graph)->getFaceLandmarkData(index);
        if (datas.empty())
        {
            *data = nullptr;
            *rect = nullptr;
            return 0;
        }
        else
        {
            auto landmark_data = (uint8_t*)malloc(datas.mLandmarks.size() * 3 * sizeof(float));
            for (int i = 0; i < datas.mLandmarks.size(); i++)
            {
                memcpy(landmark_data + i * 3 * sizeof(float), datas.mLandmarks[i].data(), 3 * sizeof(float));
            }

            auto landmark_rect = (uint8_t*)malloc(4 * sizeof(float));
            memcpy(landmark_rect, datas.mRect.data(), 4 * sizeof(float));

            *data_size = datas.mLandmarks.size() * 3;
            *data = (float*)landmark_data;
            *rect = (float*)landmark_rect;

            return datas.mRotation;
        }
    }

#ifdef __ANDROID__
    void sx_initAssetManager(JNIEnv* env, jobject context, const char* cache_path)
    {
        Singleton<mediapipe::AssetManager>::get()->InitializeFromContext(env, context, cache_path);
    }
#endif
}