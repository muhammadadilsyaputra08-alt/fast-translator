#include "jni_bridge.h"
#include "engine.h"
#include "neon_selftest.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace {

// Tracks in-flight generations so nativeCancelGeneration can signal them.
// Keyed by a monotonically increasing generation handle.
std::mutex g_cancel_mutex;
std::unordered_map<int64_t, std::shared_ptr<std::atomic<bool>>> g_cancel_flags;
std::atomic<int64_t> g_next_generation_handle{1};

JavaVM* g_jvm = nullptr;

std::string jstring_to_std_string(JNIEnv* env, jstring jstr) {
    if (jstr == nullptr) return {};
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

} // namespace

extern "C" {

jint JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL
Java_com_fasttranslator_FastModelNative_nativeLoadModel(
    JNIEnv* env, jobject /*thiz*/,
    jstring modelPath, jint backend,
    jboolean /*enableMiniCache*/, jboolean /*enableRocketKV*/, jint /*compressEvery*/) {

    std::string path = jstring_to_std_string(env, modelPath);
    auto eng = engine::FastEngine::load(path, static_cast<engine::Backend>(backend));
    if (!eng) return 0;

    // Release ownership to a raw handle the Kotlin side holds until close().
    return reinterpret_cast<jlong>(eng.release());
}

JNIEXPORT jlong JNICALL
Java_com_fasttranslator_FastModelNative_nativeLoadRawModel(
    JNIEnv* env, jobject /*thiz*/,
    jstring checkpointPath, jstring tokenizerPath, jint backend) {

    std::string ckpt = jstring_to_std_string(env, checkpointPath);
    std::string tok = jstring_to_std_string(env, tokenizerPath);
    auto eng = engine::FastEngine::load_raw(ckpt, tok, static_cast<engine::Backend>(backend));
    if (!eng) return 0;
    return reinterpret_cast<jlong>(eng.release());
}

JNIEXPORT jlong JNICALL
Java_com_fasttranslator_FastModelNative_nativeGenerate(
    JNIEnv* env, jobject /*thiz*/,
    jlong handle, jstring prompt, jstring grammar,
    jint maxTokens, jfloat temperature, jobject callback) {

    auto* eng = reinterpret_cast<engine::FastEngine*>(handle);
    if (eng == nullptr) return 0;

    engine::GenerateOptions opts;
    opts.prompt = jstring_to_std_string(env, prompt);
    opts.grammar_gbnf = jstring_to_std_string(env, grammar);
    opts.max_tokens = maxTokens;
    opts.temperature = temperature;

    // The callback object and any strings must outlive this JNI call, since
    // generation continues on a detached worker thread after we return.
    jobject callback_global = env->NewGlobalRef(callback);

    int64_t gen_handle = g_next_generation_handle.fetch_add(1);
    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard<std::mutex> lock(g_cancel_mutex);
        g_cancel_flags[gen_handle] = cancel_flag;
    }

    std::thread([eng, opts, callback_global, gen_handle, cancel_flag]() {
        JNIEnv* thread_env = nullptr;
        if (g_jvm->AttachCurrentThread(&thread_env, nullptr) != JNI_OK) {
            return;
        }

        jclass cb_class = thread_env->GetObjectClass(callback_global);
        jmethodID on_token = thread_env->GetMethodID(cb_class, "onToken", "(Ljava/lang/String;)V");
        jmethodID on_complete = thread_env->GetMethodID(cb_class, "onComplete", "()V");

        eng->generate(
            opts,
            [&](const std::string& token) {
                jstring jtoken = thread_env->NewStringUTF(token.c_str());
                thread_env->CallVoidMethod(callback_global, on_token, jtoken);
                thread_env->DeleteLocalRef(jtoken);
            },
            [&]() { return cancel_flag->load(); }
        );

        // Signal natural completion (reached EOS/max_tokens, or was
        // cancelled) so the Kotlin Flow can close itself instead of
        // waiting forever for a token that will never come.
        thread_env->CallVoidMethod(callback_global, on_complete);

        thread_env->DeleteGlobalRef(callback_global);
        {
            std::lock_guard<std::mutex> lock(g_cancel_mutex);
            g_cancel_flags.erase(gen_handle);
        }
        g_jvm->DetachCurrentThread();
    }).detach();

    return static_cast<jlong>(gen_handle);
}

JNIEXPORT void JNICALL
Java_com_fasttranslator_FastModelNative_nativeCancelGeneration(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong /*handle*/, jlong generationHandle) {

    std::lock_guard<std::mutex> lock(g_cancel_mutex);
    auto it = g_cancel_flags.find(generationHandle);
    if (it != g_cancel_flags.end()) {
        it->second->store(true);
    }
}

JNIEXPORT void JNICALL
Java_com_fasttranslator_FastModelNative_nativeFreeModel(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {

    auto* eng = reinterpret_cast<engine::FastEngine*>(handle);
    delete eng;
}

JNIEXPORT jstring JNICALL
Java_com_fasttranslator_FastModelNative_nativeVerifyNeon(
    JNIEnv* env, jobject /*thiz*/) {
    std::string report = neon_selftest::run();
    return env->NewStringUTF(report.c_str());
}

} // extern "C"
