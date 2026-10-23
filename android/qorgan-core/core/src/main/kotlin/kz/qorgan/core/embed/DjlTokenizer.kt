package kz.qorgan.core.embed

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import java.io.InputStream
import java.nio.file.Path

/**
 * The HF `tokenizers` Rust library via DJL's JNI binding: it reads the model's own tokenizer.json
 * (XLM-R SentencePiece-Unigram + precompiled normaliser), the same library the Python server uses.
 * Truncation to 512 (special tokens included) like `embed.py` and transformers.js; no padding
 * (one text per run).
 */
class DjlTokenizer private constructor(private val tokenizer: HuggingFaceTokenizer) : TokenIdEncoder, AutoCloseable {
    override fun encode(text: String): LongArray = tokenizer.encode(text).ids

    override fun close() = tokenizer.close()

    companion object {
        const val MAX_LENGTH = 512

        private val OPTIONS = mapOf(
            "addSpecialTokens" to "true",
            "truncation" to "true",
            "padding" to "false",
            "maxLength" to MAX_LENGTH.toString(),
        )

        fun fromFile(tokenizerJson: Path): DjlTokenizer = DjlTokenizer(HuggingFaceTokenizer.newInstance(tokenizerJson, OPTIONS))

        fun fromStream(tokenizerJson: InputStream): DjlTokenizer = DjlTokenizer(HuggingFaceTokenizer.newInstance(tokenizerJson, OPTIONS))
    }
}
