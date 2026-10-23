package kz.qorgan.core.model

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Small, failing-loud accessors over the JSON tree (config / weights are external input). */
internal class ConfigError(message: String) : IllegalArgumentException(message)

internal fun JsonObject.obj(key: String): JsonObject =
    (this[key] as? JsonObject) ?: throw ConfigError("missing object '$key'")

internal fun JsonObject.arr(key: String): JsonArray =
    (this[key] as? JsonArray) ?: throw ConfigError("missing array '$key'")

internal fun JsonObject.str(key: String): String =
    (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content ?: throw ConfigError("missing string '$key'")

internal fun JsonObject.num(key: String): Double =
    (this[key] as? JsonPrimitive)?.double ?: throw ConfigError("missing number '$key'")

internal fun JsonObject.int(key: String): Int =
    (this[key] as? JsonPrimitive)?.int ?: throw ConfigError("missing int '$key'")

internal fun JsonObject.numOr(key: String, default: Double): Double =
    (this[key] as? JsonPrimitive)?.double ?: default

internal fun JsonElement.strings(): List<String> = jsonArray.map { it.jsonPrimitive.content }

internal fun JsonElement.doubles(): DoubleArray = jsonArray.map { it.jsonPrimitive.double }.toDoubleArray()

internal fun JsonObject.stringMap(): Map<String, String> = mapValues { it.value.jsonPrimitive.content }

internal fun JsonObject.numberMap(): Map<String, Double> = mapValues { it.value.jsonPrimitive.double }

internal fun JsonElement.asObj(): JsonObject = jsonObject
