package kz.qorgan.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kz.qorgan.core.model.QorganConfig
import kz.qorgan.core.model.Weights
import kz.qorgan.core.session.Embedder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

/** The repository's own files, read in place (the same ones `tests_js/` reads). */
object Fixtures {
    val repo: File = File(System.getProperty("qorgan.repo") ?: error("system property qorgan.repo not set"))

    fun file(path: String): File = File(repo, path)

    fun json(path: String): JsonObject = Json.parseToJsonElement(file(path).readText()).jsonObject

    val configText: String by lazy { file("site/core/qorgan-config.json").readText() }
    val weightsText: String by lazy { file("site/models/weights.json").readText() }
    val config: QorganConfig by lazy { QorganConfig.parse(configText) }
    val weights: Weights by lazy { Weights.parse(weightsText) }
    val parity: JsonObject by lazy { json("tests_js/fixtures/parity.json") }

    fun decode(b64: String): FloatArray {
        val bytes = Base64.getDecoder().decode(b64)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buf.remaining()).also { buf.get(it) }
    }

    /** The recorded Python embeddings, keyed by exact text (like `tests_js/helpers.mjs::fixtureEmbed`). */
    val recordedEmbedder: Embedder by lazy {
        val table = parity.getValue("embeddings").jsonObject
        val cache = HashMap<String, FloatArray>()
        Embedder { texts ->
            texts.map { t ->
                cache.getOrPut(t) {
                    decode(table[t]?.jsonPrimitive?.content ?: error("no recorded embedding for: ${t.take(60)}"))
                }
            }
        }
    }
}
