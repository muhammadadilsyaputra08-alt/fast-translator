package com.fasttranslator

import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Demo screen using the final 3-step API with a single packed .fllm model
 * (INT8-quantized, produced by tools/pack_fllm.cpp — see
 * notes/FASE3_QUANTIZATION_FINDINGS.md for why INT8 was chosen over
 * ternary/BitNet: empirically near-lossless on this model, vs. ternary's
 * catastrophic quality loss even with GPTQ-style calibration).
 *
 * Push the packed model to the app's external files dir (no runtime
 * storage permission needed):
 *   adb push model.fllm /sdcard/Android/data/com.fasttranslator/files/models/model.fllm
 */
class MainActivity : ComponentActivity() {

    companion object {
        // Kalimat demo untuk tombol Generate/Benchmark (belum ada input teks di UI ini).
        private const val DEMO_SOURCE_TEXT = "Bonjour, comment ca va aujourd'hui?"

        /**
         * Template terjemahan final -- varian "B. Few-shot 2 contoh" dari hasil iterasi
         * prompt (lihat notes/TINYLLAMA_EXPORT_QUANTIZATION_REPORT.md, Bagian 11).
         * Satu-satunya gaya di antara A-D yang konsisten benar arah Prancis->Inggris
         * DAN berhenti bersih tanpa bocor ke label/teks lain. Menggantikan hardcode
         * "Once upon a time" (peninggalan test stories15M, bukan task terjemahan).
         */
        private fun buildTranslationPrompt(sourceText: String): String =
            "French: Je m'appelle Marie.\n" +
            "English: My name is Marie.\n\n" +
            "French: Il fait beau aujourd'hui.\n" +
            "English: The weather is nice today.\n\n" +
            "French: $sourceText\n" +
            "English:"

        // Model biasa echo balik label "English:" sebagai "English: " di awal output --
        // strip prefix ini sekali sebelum ditampilkan ke user.
        private fun stripPromptArtifact(text: String): String =
            text.trimStart(':', ' ', '\n')
    }

    private lateinit var statusView: TextView
    private lateinit var outputView: TextView
    private var model: FastModel? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }

        statusView = TextView(this).apply {
            text = "FastTranslator Engine — demo\nTap Load after pushing model.fllm"
        }
        outputView = TextView(this).apply { text = "" }
        val loadButton = Button(this).apply { text = "Load model" }
        val generateButton = Button(this).apply { text = "Generate" }
        val generateJsonButton = Button(this).apply { text = "Generate (JSON grammar)" }
        val benchmarkButton = Button(this).apply { text = "Benchmark (measure real tok/s)" }
        val verifyNeonButton = Button(this).apply { text = "Verify NEON correctness" }
        val benchmarkThreadsButton = Button(this).apply { text = "Benchmark thread counts" }
        val benchmarkVulkanButton = Button(this).apply { text = "Benchmark Vulkan vs CPU" }

        loadButton.setOnClickListener { onLoadClicked() }
        generateButton.setOnClickListener { onGenerateClicked(Grammar.None) }
        generateJsonButton.setOnClickListener { onGenerateClicked(Grammar.JSON) }
        benchmarkButton.setOnClickListener { onBenchmarkClicked() }
        verifyNeonButton.setOnClickListener {
            statusView.text = FastModel.verifyNeon()
        }
        benchmarkThreadsButton.setOnClickListener {
            // Unlike verifyNeon() (a few small synthetic matmuls, fast enough
            // for the main thread), this sweeps up to 6 thread-count configs
            // x 30 iterations of a full 2048x2048 matmul -- easily several
            // seconds on a mid-range SoC, so it MUST run off the main thread
            // or it risks an ANR exactly like the model-loading bug we fixed
            // earlier (see notes/TINYLLAMA_EXPORT_QUANTIZATION_REPORT.md).
            statusView.text = "Benchmarking thread counts (this takes a few seconds)..."
            lifecycleScope.launch {
                val report = withContext(Dispatchers.IO) { FastModel.benchmarkThreads() }
                statusView.text = report
            }
        }
        benchmarkVulkanButton.setOnClickListener {
            // Same ANR concern as benchmarkThreadsButton: Vulkan instance/
            // device setup + 30 GPU dispatch round-trips is not instant.
            statusView.text = "Benchmarking Vulkan vs CPU (this takes a few seconds)..."
            lifecycleScope.launch {
                val report = withContext(Dispatchers.IO) { FastModel.benchmarkVulkan() }
                statusView.text = report
            }
        }

        layout.addView(statusView)
        layout.addView(loadButton)
        layout.addView(generateButton)
        layout.addView(generateJsonButton)
        layout.addView(benchmarkButton)
        layout.addView(verifyNeonButton)
        layout.addView(benchmarkThreadsButton)
        layout.addView(benchmarkVulkanButton)
        layout.addView(outputView)
        setContentView(layout)

        // Auto-load on startup instead of waiting for a manual tap.
        onLoadClicked()
    }

    private fun modelFile(): File = File(getExternalFilesDir("models"), "model.fllm")

    private fun onLoadClicked() {
        val file = modelFile()
        if (!file.exists()) {
            statusView.text = "Model not found at:\n${file.absolutePath}\n\n" +
                "Push it with:\nadb push model.fllm ${file.absolutePath}"
            return
        }

        statusView.text = "Loading model..."
        lifecycleScope.launch {
            try {
                model?.close()
                // FastModel.load() is a plain blocking call (disk read + native
                // parse/alloc of a 1.5GB+ file) -- it MUST NOT run on the coroutine's
                // default dispatcher (Dispatchers.Main via lifecycleScope), or it
                // freezes the UI thread for the whole load and triggers Android's
                // ANR watchdog, which the user experiences as the app crashing.
                val loaded = withContext(Dispatchers.IO) {
                    FastModel.load(
                        modelPath = file.absolutePath,
                        backend = Backend.QNN,
                        kvCacheConfig = KVCacheConfig(enableMiniCache = true)
                    )
                }
                model = loaded
                statusView.text = "Model loaded OK from:\n${file.absolutePath}"
            } catch (e: IllegalStateException) {
                statusView.text = "Load failed: ${e.message}"
            } catch (e: OutOfMemoryError) {
                // Best-effort: a native allocation failure inside the JNI call
                // usually kills the whole process before Kotlin ever sees this,
                // but Dalvik-side OOMs (e.g. while marshalling the path string)
                // are catchable, so handle it rather than letting it propagate.
                statusView.text = "Out of memory while loading model. " +
                    "Close other apps and try again, or restart the device."
            }
        }
    }

    private fun onGenerateClicked(grammar: Grammar) {
        val currentModel = model
        if (currentModel == null) {
            statusView.text = "Load the model first."
            return
        }

        outputView.text = ""
        lifecycleScope.launch {
            try {
                val buffer = StringBuilder()
                var artifactStripped = false
                currentModel.generate(
                    prompt = buildTranslationPrompt(DEMO_SOURCE_TEXT),
                    grammar = grammar,
                    maxTokens = if (grammar == Grammar.None) 40 else 25 // grammar sampling is slower (~10x, full-vocab mask per step)
                ).collect { token ->
                    if (!artifactStripped) {
                        buffer.append(token)
                        val cleaned = stripPromptArtifact(buffer.toString())
                        if (cleaned.isNotEmpty()) {
                            outputView.text = cleaned
                            artifactStripped = true
                        }
                    } else {
                        outputView.append(token)
                    }
                }
                statusView.text = "Generation complete."
            } catch (e: Exception) {
                statusView.text = "Generate failed: ${e.message}"
            }
        }
    }

    override fun onDestroy() {
        model?.close()
        super.onDestroy()
    }

    /**
     * Measures real on-device generation speed (tok/s) for the CPU
     * reference backend actually running right now — NOT NPU-accelerated.
     * `Backend.QNN` passed to FastModel.load() is currently just a label;
     * there is no QNN/HTP kernel implementation yet (see
     * notes/NPU_BENCHMARK_STATUS.md), so this number is the honest
     * baseline any future real NPU backend should be compared against.
     */
    private fun onBenchmarkClicked() {
        val currentModel = model
        if (currentModel == null) {
            statusView.text = "Load the model first."
            return
        }

        outputView.text = ""
        statusView.text = "Benchmarking..."
        lifecycleScope.launch {
            try {
                var tokenCount = 0
                val buffer = StringBuilder()
                var artifactStripped = false
                val startMs = System.currentTimeMillis()
                currentModel.generate(
                    prompt = buildTranslationPrompt(DEMO_SOURCE_TEXT),
                    grammar = Grammar.None,
                    maxTokens = 80
                ).collect { token ->
                    tokenCount++
                    if (!artifactStripped) {
                        buffer.append(token)
                        val cleaned = stripPromptArtifact(buffer.toString())
                        if (cleaned.isNotEmpty()) {
                            outputView.text = cleaned
                            artifactStripped = true
                        }
                    } else {
                        outputView.append(token)
                    }
                }
                val elapsedMs = System.currentTimeMillis() - startMs
                val tokPerSec = if (elapsedMs > 0) tokenCount * 1000.0 / elapsedMs else 0.0
                statusView.text = "Benchmark: %d tokens in %d ms = %.2f tok/s (CPU reference, no NPU)"
                    .format(tokenCount, elapsedMs, tokPerSec)
            } catch (e: Exception) {
                statusView.text = "Benchmark failed: ${e.message}"
            }
        }
    }
}
