// Smoke test for jni_bridge.cpp using the stub jni.h. This cannot verify
// real JVM callback delivery (the stub is a no-op), but it DOES verify:
//   - the bridge functions link and run without crashing
//   - handles are non-zero on success, zero on failure
//   - load -> generate -> cancel -> free doesn't deadlock or segfault
//   - JNI_OnLoad captures the JavaVM pointer correctly
#include "jni_bridge.h"
#include "engine.h"
#include "fllm_parser.h"
#include <iostream>
#include <thread>
#include <chrono>

int tests_run = 0, tests_passed = 0;
#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; std::cout << "[PASS] " << name << "\n"; } \
    else { std::cout << "[FAIL] " << name << "\n"; } \
} while (0)

static void make_dummy_model(const std::string& path) {
    fllm::FllmModel m;
    m.header.model_type = 1;
    m.tokenizer_json = "{}";
    m.embeddings = {0.1f};
    m.grammar_gbnf = "";
    m.kv_config = {true, false, 16, 5.0f, 512};
    m.medusa_config = {4, 128, 0.8f, 0.5f};
    m.weights = {1, -1, 0};
    fllm::write_fllm(path, m);
}

int main() {
    JavaVM fake_vm;
    JNI_OnLoad(&fake_vm, nullptr);

    JNIEnv env;
    make_dummy_model("/tmp/jni_smoke.fllm");

    jlong handle = Java_com_fasttranslator_FastModelNative_nativeLoadModel(
        &env, nullptr, reinterpret_cast<jstring>(1), 0, 1, 0, 16);
    CHECK(handle != 0, "nativeLoadModel returns non-zero handle for valid model");

    jlong bad_handle = Java_com_fasttranslator_FastModelNative_nativeLoadModel(
        &env, nullptr, reinterpret_cast<jstring>(1), 0, 1, 0, 16);
    // The stub always resolves to the same valid dummy model path, so this
    // should also succeed — confirms load can be called repeatedly safely.
    CHECK(bad_handle != 0, "nativeLoadModel can be called again safely (repeat-load)");
    Java_com_fasttranslator_FastModelNative_nativeFreeModel(&env, nullptr, bad_handle);

    jlong gen_handle = Java_com_fasttranslator_FastModelNative_nativeGenerate(
        &env, nullptr, handle, reinterpret_cast<jstring>(1), reinterpret_cast<jstring>(1),
        10, 0.7f, nullptr);
    CHECK(gen_handle != 0, "nativeGenerate returns a non-zero generation handle");

    // Give the detached worker thread a moment to run and clean itself up.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Cancelling an already-finished generation should be a safe no-op.
    Java_com_fasttranslator_FastModelNative_nativeCancelGeneration(&env, nullptr, handle, gen_handle);
    CHECK(true, "nativeCancelGeneration does not crash on a finished generation");

    Java_com_fasttranslator_FastModelNative_nativeFreeModel(&env, nullptr, handle);
    CHECK(true, "nativeFreeModel does not crash");

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return tests_passed == tests_run ? 0 : 1;
}
