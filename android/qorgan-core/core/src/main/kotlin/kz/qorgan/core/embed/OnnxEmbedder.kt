package kz.qorgan.core.embed

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kz.qorgan.core.session.Embedder
import java.nio.LongBuffer
import kotlin.math.sqrt

/** Text -> token ids with special tokens, truncated to the model's 512 (the HF `tokenizers` recipe). */
fun interface TokenIdEncoder {
    fun encode(text: String): LongArray
}

/**
 * multilingual-e5-base int8 ONNX -> attention-masked mean pooling -> L2 normalisation: the recipe of
 * `qorgan/classifier/embed.py::OnnxEmbedder` and `site/core/embed-worker.js`. ONE text per graph
 * run, deliberately: the int8 graph is dynamically quantised, so a batched embedding would depend
 * on its neighbours. Runs entirely in-process; nothing is written anywhere.
 */
class OnnxEmbedder(
    private val session: OrtSession,
    private val tokenizer: TokenIdEncoder,
    private val prefix: String,
) : Embedder, AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val inputNames = session.inputNames

    override fun embed(texts: List<String>): List<FloatArray> = texts.map { embedOne(prefix + it) }

    private fun embedOne(text: String): FloatArray {
        val ids = tokenizer.encode(text)
        val shape = longArrayOf(1, ids.size.toLong())
        val tensors = LinkedHashMap<String, OnnxTensor>()
        try {
            tensors["input_ids"] = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
            tensors["attention_mask"] = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size) { 1 }), shape)
            if ("token_type_ids" in inputNames) {
                tensors["token_type_ids"] = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size)), shape)
            }
            session.run(tensors).use { result ->
                @Suppress("UNCHECKED_CAST")
                val hidden = (result.get(0).value as Array<Array<FloatArray>>)[0]
                return meanPoolNormalize(hidden)
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    override fun close() = session.close()

    companion object {
        private const val NORM_EPS = 1e-12

        /** Mean over tokens (every token is real: one text per run, no padding), then L2. */
        fun meanPoolNormalize(hidden: Array<FloatArray>): FloatArray {
            val dim = hidden[0].size
            val pooled = FloatArray(dim)
            for (d in 0 until dim) {
                var acc = 0.0
                for (row in hidden) acc += row[d]
                pooled[d] = (acc / hidden.size).toFloat()
            }
            var sq = 0.0
            for (v in pooled) sq += v.toDouble() * v
            val norm = maxOf(sqrt(sq), NORM_EPS)
            return FloatArray(dim) { (pooled[it] / norm).toFloat() }
        }

        /** Default CPU session (same execution provider as the server). */
        fun createSession(modelBytesOrPath: Any, threads: Int = 0): OrtSession {
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply { if (threads > 0) setIntraOpNumThreads(threads) }
            return when (modelBytesOrPath) {
                is String -> env.createSession(modelBytesOrPath, options)
                is ByteArray -> env.createSession(modelBytesOrPath, options)
                else -> throw IllegalArgumentException("model must be a file path or bytes")
            }
        }
    }
}

/**
 * In-memory cache in front of an embedder: `score()` re-embeds every utterance of the rolling window
 * on each turn, and an embedding is a pure function of its text. Bounded LRU; never persisted.
 */
class CachingEmbedder(private val inner: Embedder, private val capacity: Int = 256) : Embedder {
    private val cache = object : LinkedHashMap<String, FloatArray>(capacity, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, FloatArray>) = size > capacity
    }

    @Synchronized
    override fun embed(texts: List<String>): List<FloatArray> {
        val missing = texts.filter { it !in cache }.distinct()
        if (missing.isNotEmpty()) inner.embed(missing).forEachIndexed { i, v -> cache[missing[i]] = v }
        return texts.map { cache.getValue(it) }
    }

    @Synchronized
    fun clear() = cache.clear()
}
