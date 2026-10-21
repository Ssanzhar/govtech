package kz.qorgan.core.android

import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kz.qorgan.core.gate.RuntimeGate
import kz.qorgan.core.session.Embedder
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The runtime gate ON THE PHONE (not run yet: needs a device). Push the embedder once:
 *   adb push site/models/Xenova/multilingual-e5-base/onnx/model_quantized.onnx \
 *            /sdcard/Android/data/kz.qorgan.core.test/files/model_quantized.onnx
 *   adb push site/models/Xenova/multilingual-e5-base/tokenizer.json \
 *            /sdcard/Android/data/kz.qorgan.core.test/files/tokenizer.json
 * then `./gradlew :aar:connectedAndroidTest` and read `adb logcat -s QorganGate`.
 * Reports decisions vs Python AND vs the browser WASM embeddings the heads were trained on, the
 * per-transcript latency and the memory after loading the model.
 */
@RunWith(AndroidJUnit4::class)
class DeviceRuntimeGateTest {
    @Test
    fun androidOrtReproducesTheGateDecisions() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val dir = context.getExternalFilesDir(null) ?: error("no external files dir")
        val model = File(dir, "model_quantized.onnx")
        val tokenizer = File(dir, "tokenizer.json")
        assumeTrue("push the model to $dir first (see KDoc)", model.isFile && tokenizer.isFile)

        val t0 = System.nanoTime()
        val core = QorganAndroid.load(context, model, tokenizer)
        val loadMs = (System.nanoTime() - t0) / 1e6
        val memory = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
        val assets = instrumentation.context.assets
        val gate = assets.open("gate/runtime_gate.json").bufferedReader().use { it.readText() }
        val browser = assets.open("gate/runtime_gate_browser.f32").use { it.readBytes() }
        val embedder: Embedder = core.embedder
        val result = RuntimeGate.run(gate, browser, core.weights, core.config, embedder)
        val line = "load=${"%.0f".format(loadMs)} ms, PSS after load=${memory.totalPss / 1024} MB, " +
            "native heap=${Debug.getNativeHeapAllocatedSize() / 1_048_576} MB; ${result.summary()}"
        Log.i("QorganGate", line)
        result.vsPython.flips.forEach { Log.i("QorganGate", "flip vs Python: $it") }
        result.vsBrowser?.flips?.forEach { Log.i("QorganGate", "flip vs browser: $it") }
        assertTrue(line, result.passes)
    }
}
