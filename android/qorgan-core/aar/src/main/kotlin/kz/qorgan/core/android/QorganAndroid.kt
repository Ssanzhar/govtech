package kz.qorgan.core.android

import android.content.Context
import kz.qorgan.core.QorganCore
import kz.qorgan.core.embed.CachingEmbedder
import kz.qorgan.core.embed.DjlTokenizer
import kz.qorgan.core.embed.OnnxEmbedder
import java.io.File

/**
 * Android factory for [QorganCore]. Config + head weights come from the AAR's assets (copied from
 * site/ at build time). The embedder files are either bundled assets (`-PqorganBundleModel=true`)
 * or files the host app already has on disk (e.g. delivered by eGov Mobile on demand); in both
 * cases the model runs in-process and no text or audio leaves the device.
 */
object QorganAndroid {
    private const val ASSET_DIR = "qorgan"
    private const val MODEL_ASSET_DIR = "$ASSET_DIR/multilingual-e5-base"

    /** Load with the embedder files at [modelFile] / [tokenizerFile] (host-provided). */
    fun load(context: Context, modelFile: File, tokenizerFile: File, threads: Int = 0): QorganCore {
        require(modelFile.isFile) { "embedder model not found: $modelFile" }
        require(tokenizerFile.isFile) { "tokenizer not found: $tokenizerFile" }
        val session = OnnxEmbedder.createSession(modelFile.absolutePath, threads)
        return build(context, session, DjlTokenizer.fromFile(tokenizerFile.toPath()))
    }

    /** Load with the embedder bundled in the AAR's assets (demo builds only: +266 MB). */
    fun loadBundled(context: Context, threads: Int = 0): QorganCore {
        val assets = context.assets
        // ORT needs a path or bytes; copy once into the app's no-backup dir (model files only, no user data).
        val target = File(context.noBackupFilesDir, "qorgan-model.onnx")
        if (!target.isFile) {
            assets.open("$MODEL_ASSET_DIR/onnx/model_quantized.onnx").use { input ->
                target.outputStream().use { input.copyTo(it) }
            }
        }
        val tokenizer = assets.open("$MODEL_ASSET_DIR/tokenizer.json").use { DjlTokenizer.fromStream(it) }
        return build(context, OnnxEmbedder.createSession(target.absolutePath, threads), tokenizer)
    }

    private fun build(context: Context, session: ai.onnxruntime.OrtSession, tokenizer: DjlTokenizer): QorganCore {
        val config = readAsset(context, "$ASSET_DIR/qorgan-config.json")
        val weights = readAsset(context, "$ASSET_DIR/weights.json")
        val prefix = kz.qorgan.core.model.QorganConfig.parse(config).embedder.prefix
        return QorganCore.fromJson(config, weights, CachingEmbedder(OnnxEmbedder(session, tokenizer, prefix)))
    }

    private fun readAsset(context: Context, path: String): String =
        context.assets.open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }
}
