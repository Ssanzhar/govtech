package kz.qorgan.core.model

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject

/** `site/models/weights.json` (layout: `qorgan/classifier/web_bundle.py`), the shipped model. */
data class Weights(
    val labelSpace: List<String>,
    val embeddingDim: Int,
    val cueMatcherVersion: Int,
    val riskMembers: List<RiskMember>,
    val tacticModels: Map<String, LinearModel>,
    val tacticThreshold: Double,
    val tacticThresholds: Map<String, Double>,
    val cues: Map<String, List<String>>,
    val reassurance: ReassurancePatterns,
) {
    companion object {
        private const val FORMAT_VERSION = 1
        private const val RISK_HEAD_TYPE = "calibrated_lr_sigmoid_mean"

        fun parse(text: String): Weights = fromJson(Json.parseToJsonElement(text).jsonObject)

        fun fromJson(root: JsonObject): Weights {
            val format = root.int("format_version")
            if (format != FORMAT_VERSION) throw ConfigError("unsupported weights format $format")
            val lexicon = (root["lexicon"] as? JsonObject)
                ?: throw ConfigError("the on-device scorer needs a hybrid bundle (lexicon block)")
            val riskHead = root.obj("risk_head")
            val type = riskHead.str("type")
            if (type != RISK_HEAD_TYPE) throw ConfigError("unsupported risk head type: $type")
            val tacticHead = root.obj("tactic_head")
            val reassurance = lexicon.obj("reassurance")
            return Weights(
                labelSpace = root.arr("label_space").strings(),
                embeddingDim = root.int("embedding_dim"),
                cueMatcherVersion = root.int("cue_matcher_version"),
                riskMembers = riskHead.arr("members").map { m ->
                    val o = m.asObj()
                    RiskMember(o.arr("coef").doubles(), o.num("intercept"), o.num("calib_a"), o.num("calib_b"))
                },
                tacticModels = tacticHead.obj("models").mapValues { (_, v) ->
                    val o = v.asObj()
                    LinearModel(o.arr("coef").doubles(), o.num("intercept"))
                },
                tacticThreshold = tacticHead.num("threshold"),
                tacticThresholds = (tacticHead["thresholds"] as? JsonObject)?.numberMap() ?: emptyMap(),
                cues = lexicon.obj("cues").mapValues { it.value.strings() },
                reassurance = ReassurancePatterns(
                    sensitiveTerms = reassurance.arr("sensitive_terms").strings(),
                    reassuranceTerms = reassurance.arr("reassurance_terms").strings(),
                    windowChars = reassurance.int("window_chars"),
                ),
            )
        }
    }
}

class RiskMember(val coef: DoubleArray, val intercept: Double, val calibA: Double, val calibB: Double)

class LinearModel(val coef: DoubleArray, val intercept: Double)

data class ReassurancePatterns(val sensitiveTerms: List<String>, val reassuranceTerms: List<String>, val windowChars: Int)
