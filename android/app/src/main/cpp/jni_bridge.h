#pragma once
#include <jni.h>

// JNI entry points backing com.fasttranslator.FastModelNative (a top-level
// Kotlin object, not a companion object, so the mangled JNI symbol names
// stay simple: Java_com_fasttranslator_FastModelNative_<method>).
extern "C" {

// Called by the JVM when the native library is loaded via System.loadLibrary.
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved);

JNIEXPORT jlong JNICALL
Java_com_fasttranslator_FastModelNative_nativeLoadModel(
    JNIEnv* env, jobject thiz,
    jstring modelPath, jint backend,
    jboolean enableMiniCache, jboolean enableRocketKV, jint compressEvery);

// Loads a real transformer checkpoint (llama2.c format, e.g. stories15M.bin)
// + its tokenizer.bin directly, bypassing the .fllm container. Used while
// the .fllm packer for real weights (Fase 4 work) doesn't exist yet.
JNIEXPORT jlong JNICALL
Java_com_fasttranslator_FastModelNative_nativeLoadRawModel(
    JNIEnv* env, jobject thiz,
    jstring checkpointPath, jstring tokenizerPath, jint backend);

// Starts generation on a background thread; tokens are delivered via
// callback.onToken(String) as they're produced. Returns a generation handle
// that can be passed to nativeCancelGeneration to stop it early.
JNIEXPORT jlong JNICALL
Java_com_fasttranslator_FastModelNative_nativeGenerate(
    JNIEnv* env, jobject thiz,
    jlong handle, jstring prompt, jstring grammar,
    jint maxTokens, jfloat temperature, jobject callback);

JNIEXPORT void JNICALL
Java_com_fasttranslator_FastModelNative_nativeCancelGeneration(
    JNIEnv* env, jobject thiz, jlong handle, jlong generationHandle);

JNIEXPORT void JNICALL
Java_com_fasttranslator_FastModelNative_nativeFreeModel(
    JNIEnv* env, jobject thiz, jlong handle);

// Runs the on-device NEON-vs-scalar correctness self-test (see
// core/src/neon_selftest.cpp) and returns a human-readable report string.
// Does not require a loaded model.
JNIEXPORT jstring JNICALL
Java_com_fasttranslator_FastModelNative_nativeVerifyNeon(
    JNIEnv* env, jobject thiz);

}
