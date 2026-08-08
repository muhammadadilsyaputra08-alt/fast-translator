package com.fasttranslator

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.channels.awaitClose

/**
 * Public 3-step API:
 *   1. val model = FastModel.load(path, backend, kvCacheConfig)
 *   2. model.generate(prompt, grammar, maxTokens, temperature).collect { token -> ... }
 *   3. model.close()
 */
class FastModel private constructor(private var nativeHandle: Long) {

    /** Called from the native worker thread: once per token, and once
     *  (onComplete) when generation naturally finishes -- without this
     *  second signal, the Kotlin Flow never closes on its own and
     *  collect{} hangs forever even after all tokens have arrived. */
    interface TokenCallback {
        fun onToken(token: String)
        fun onComplete()
    }

    private val isClosed: Boolean
        get() = nativeHandle == 0L

    /**
     * Streams generated tokens as a cold [Flow]. Collection runs on
     * [Dispatchers.IO]; cancelling collection (e.g. the coroutine scope
     * being cancelled) signals the native side to stop generating early
     * via [FastModelNative.nativeCancelGeneration].
     */
    fun generate(
        prompt: String,
        grammar: Grammar = Grammar.None,
        maxTokens: Int = 128,
        temperature: Float = 0.7f
    ): Flow<String> = callbackFlow {
        check(!isClosed) { "FastModel is closed; create a new instance with FastModel.load()." }

        val callback = object : TokenCallback {
            override fun onToken(token: String) { trySend(token) }
            override fun onComplete() { close() } // normal completion: lets collect{} return
        }
        val generationHandle = FastModelNative.nativeGenerate(
            nativeHandle, prompt, grammar.toGbnfOrNull(), maxTokens, temperature, callback
        )

        awaitClose {
            if (!isClosed) {
                FastModelNative.nativeCancelGeneration(nativeHandle, generationHandle)
            }
        }
    }.flowOn(Dispatchers.IO)

    /** Releases native resources (model weights, KV cache, NPU context). Idempotent. */
    fun close() {
        if (!isClosed) {
            FastModelNative.nativeFreeModel(nativeHandle)
            nativeHandle = 0L
        }
    }

    protected fun finalize() {
        // Safety net only — callers should still call close() explicitly,
        // since finalizers run at an unpredictable time (or not at all).
        close()
    }

    companion object {
        /** Runs the on-device NEON-vs-scalar matmul correctness self-test.
         *  Does not require a model to be loaded. See core/src/neon_selftest.cpp. */
        fun verifyNeon(): String = FastModelNative.nativeVerifyNeon()

        /**
         * Loads a `.fllm` model from [modelPath]. Throws [IllegalStateException]
         * if the file is missing, malformed, or fails checksum validation.
         */
        fun load(
            modelPath: String,
            backend: Backend = Backend.QNN,
            kvCacheConfig: KVCacheConfig = KVCacheConfig()
        ): FastModel {
            val handle = FastModelNative.nativeLoadModel(
                modelPath,
                backend.ordinal,
                kvCacheConfig.enableMiniCache,
                kvCacheConfig.enableRocketKV,
                kvCacheConfig.compressEvery
            )
            check(handle != 0L) { "Failed to load model from $modelPath (missing file or checksum mismatch)" }
            return FastModel(handle)
        }

        /**
         * Loads a real transformer checkpoint (llama2.c format, e.g.
         * stories15M.bin) directly, together with its tokenizer.bin.
         * This bypasses the .fllm container and runs genuine forward-pass
         * inference — use this to test real text generation before the
         * .fllm packer for trained weights exists.
         */
        fun loadRaw(
            checkpointPath: String,
            tokenizerPath: String,
            backend: Backend = Backend.QNN
        ): FastModel {
            val handle = FastModelNative.nativeLoadRawModel(checkpointPath, tokenizerPath, backend.ordinal)
            check(handle != 0L) { "Failed to load raw checkpoint from $checkpointPath / $tokenizerPath" }
            return FastModel(handle)
        }
    }
}
