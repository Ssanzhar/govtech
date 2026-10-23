package kz.qorgan.core

import kz.qorgan.core.gate.RuntimeGate
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * The runtime gate (ADR D32) for the JVM runtime, with the thresholds of
 * `tests_js/integration/browser_gate.test.mjs` (flips <= 3 %, p95 |Δrisk| < 0.15, tag Jaccard >= 0.9)
 * against Python's verdicts, plus the comparison with the captured browser (WASM) embeddings -- the
 * runtime the shipped heads were trained on (ADR D33). Opt-in (`-PruntimeGate=true`): ~200 int8
 * e5-base runs. The phone version is `aar/src/androidTest/.../DeviceRuntimeGateTest.kt`.
 */
class RuntimeGateTest {
    @Test
    fun jvmEmbeddingsReproducePythonDecisionsOnThe200CaseGate() {
        assumeTrue("opt in with -PruntimeGate=true", System.getProperty("qorgan.runtimeGate") == "true")
        assumeTrue("self-hosted model not downloaded", EmbedderSupport.available)
        val browser = Fixtures.file("tests_js/fixtures/runtime_gate_browser.f32").takeIf { it.isFile }?.readBytes()
        val result = RuntimeGate.run(
            Fixtures.file("tests_js/fixtures/runtime_gate.json").readText(),
            browser,
            Fixtures.weights,
            Fixtures.config,
            EmbedderSupport.embedder,
        )
        println("RuntimeGate (JVM onnxruntime ${ai.onnxruntime.OrtEnvironment.getEnvironment().version}): ${result.summary()}")
        result.vsPython.flips.forEach { println("  flip vs Python: $it") }
        result.vsBrowser?.flips?.forEach { println("  flip vs browser: $it") }
        assertTrue(result.summary(), result.passes)
    }
}
