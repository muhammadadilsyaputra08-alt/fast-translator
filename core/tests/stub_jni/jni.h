// TEST-ONLY STUB. This is NOT part of the shipped project — the real
// Android NDK provides the actual jni.h with full implementations.
// This stub exists purely so jni_bridge.cpp's syntax/types can be checked
// with a plain g++ in an environment without the NDK installed.
#pragma once
#include <cstdint>
#include <string>
#include <cstring>

using jint = int32_t;
using jlong = int64_t;
using jboolean = uint8_t;
using jfloat = float;
using jobject = void*;
using jstring = struct _jstring*;
using jclass = struct _jclass*;
using jmethodID = struct _jmethodID*;

constexpr jint JNI_OK = 0;
constexpr jint JNI_VERSION_1_6 = 0x00010006;

#define JNIEXPORT
#define JNICALL

struct JNIEnv {
    const char* GetStringUTFChars(jstring, jboolean*) { return "/tmp/jni_smoke.fllm"; }
    void ReleaseStringUTFChars(jstring, const char*) {}
    jclass GetObjectClass(jobject) { return nullptr; }
    jmethodID GetMethodID(jclass, const char*, const char*) { return nullptr; }
    void CallVoidMethod(jobject, jmethodID, ...) {}
    jstring NewStringUTF(const char*) { return nullptr; }
    void DeleteLocalRef(jstring) {}
    jobject NewGlobalRef(jobject o) { return o; }
    void DeleteGlobalRef(jobject) {}
};

struct JavaVM {
    jint AttachCurrentThread(JNIEnv** env, void*) {
        static JNIEnv stub_env;
        *env = &stub_env;
        return JNI_OK;
    }
    jint DetachCurrentThread() { return JNI_OK; }
};
