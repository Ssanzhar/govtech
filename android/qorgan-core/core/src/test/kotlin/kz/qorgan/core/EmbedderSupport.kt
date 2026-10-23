package kz.qorgan.core

import kz.qorgan.core.embed.DjlTokenizer
import kz.qorgan.core.embed.OnnxEmbedder
import java.io.File

/** Loads the self-hosted model in place (site/models/...; `scripts/deploy_bootstrap.py` fetches it). */
object EmbedderSupport {
    val modelDir: File get() = Fixtures.file("site/models/${Fixtures.config.embedder.modelId}")
    val modelFile: File get() = File(modelDir, "onnx/model_quantized.onnx")
    val tokenizerFile: File get() = File(modelDir, "tokenizer.json")
    val available: Boolean get() = modelFile.isFile && tokenizerFile.isFile

    val tokenizer: DjlTokenizer by lazy { DjlTokenizer.fromFile(tokenizerFile.toPath()) }

    val embedder: OnnxEmbedder by lazy {
        val t0 = System.nanoTime()
        val session = OnnxEmbedder.createSession(modelFile.absolutePath)
        println("ONNX session (onnxruntime JVM) loaded in ${(System.nanoTime() - t0) / 1_000_000} ms; inputs=${session.inputNames}")
        OnnxEmbedder(session, tokenizer, Fixtures.config.embedder.prefix)
    }

    fun cosine(a: FloatArray, b: FloatArray): Double {
        var s = 0.0
        for (i in a.indices) s += a[i].toDouble() * b[i]
        return s
    }
}
