package com.fasttranslator

import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
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

        loadButton.setOnClickListener { onLoadClicked() }
        generateButton.setOnClickListener { onGenerateClicked(Grammar.None) }
        generateJsonButton.setOnClickListener { onGenerateClicked(Grammar.JSON) }
        benchmarkButton.setOnClickListener { onBenchmarkClicked() }
        verifyNeonButton.setOnClickListener {
            statusView.text = FastModel.verifyNeon()
        }

        layout.addView(statusView)
        layout.addView(loadButton)
        layout.addView(generateButton)
        layout.addView(generateJsonButton)
        layout.addView(benchmarkButton)
        layout.addView(verifyNeonButton)
        layout.addView(outputView)
        setContentView(layout)
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
                model = FastModel.load(
                    modelPath = file.absolutePath,
                    backend = Backend.QNN,
                    kvCacheConfig = KVCacheConfig(enableMiniCache = true)
                )
                statusView.text = "Model loaded OK from:\n${file.absolutePath}"
            } catch (e: IllegalStateException) {
                statusView.text = "Load failed: ${e.message}"
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
                currentModel.generate(
                    prompt = "Once upon a time",
                    grammar = grammar,
                    maxTokens = if (grammar == Grammar.None) 60 else 25 // grammar sampling is slower (~10x, full-vocab mask per step)
                ).collect { token ->
                    outputView.append(token)
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
                val startMs = System.currentTimeMillis()
                currentModel.generate(
                    prompt = "Once upon a time",
                    grammar = Grammar.None,
                    maxTokens = 80
                ).collect { token ->
                    tokenCount++
                    outputView.append(token)
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
