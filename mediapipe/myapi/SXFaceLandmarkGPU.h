//
// Created by carrot hu on 2021/3/12.
//

#ifndef MEDIAPIPE_SXFACELANDMARKGPU_H
#define MEDIAPIPE_SXFACELANDMARKGPU_H

#include "SXMediapipeCommon.h"

#if __ANDROID__
#include <jni.h>
#endif

extern "C"
{
    void* SXMEDIAPIPE_API sx_createFaceLandmarkGpuGraph(bool bottomLeft, int maxFaces);

    void SXMEDIAPIPE_API sx_destroyFaceLandmarkGpuGraph(void* graph);

    bool SXMEDIAPIPE_API sx_startFaceGraph(void* graph, void* sharedContext);

    bool SXMEDIAPIPE_API sx_processTexture(void* graph, unsigned texture, int width, int height);

    bool SXMEDIAPIPE_API sx_processPixelbuffer(void* graph, void* pixelbuffer, int width, int height);

    bool SXMEDIAPIPE_API sx_stopFaceGraph(void* graph);

    int SXMEDIAPIPE_API sx_getFaceNum(void* graph);

    float SXMEDIAPIPE_API sx_getFaceLandmarkData(void* graph, int index, float** data, int* data_size, float** rect);

#ifdef __ANDROID__
    void SXMEDIAPIPE_API sx_initAssetManager(JNIEnv* env, jobject context, const char* cache_path);
#endif
}

#endif  // MEDIAPIPE_SXFACELANDMARKGPU_H
