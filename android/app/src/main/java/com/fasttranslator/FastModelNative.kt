package com.fasttranslator

/**
 * Raw JNI entry points implemented in jni_bridge.cpp. This is a top-level
 * `object`, not a companion object, so the mangled native symbol names stay
 * simple (Java_com_fasttranslator_FastModelNative_xxx) instead of getting a
 * "$Companion" suffix.
 *
 * Not intended to be called directly by app code — use [FastModel] instead.
 */
internal object FastModelNative {
    init {
        System.loadLibrary("fastai")
    }

    external fun nativeLoadModel(
        modelPath: String,
        backend: Int,
        enableMiniCache: Boolean,
        enableRocketKV: Boolean,
        compressEvery: Int
    ): Long

    external fun nativeLoadRawModel(
        checkpointPath: String,
        tokenizerPath: String,
        backend: Int
    ): Long

    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        grammar: String?,
        maxTokens: Int,
        temperature: Float,
        callback: FastModel.TokenCallback
    ): Long

    external fun nativeCancelGeneration(handle: Long, generationHandle: Long)

    external fun nativeFreeModel(handle: Long)

    external fun nativeVerifyNeon(): String

    external fun nativeBenchmarkThreads(): String

    external fun nativeBenchmarkVulkan(): String
}
