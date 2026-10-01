package io.github.xororz.localdream.data

import java.io.File
import org.json.JSONObject

/**
 * Frozen application-side contract for converted DreamLite components.
 *
 * Shapes are emitted by the converter from the reference graphs and validated
 * here before any accelerator runtime is opened.
 */
object DreamLiteAbi {
    const val VERSION = 1
    const val MANIFEST = "dreamlite_abi.json"
    const val EXPECTED_STEPS = 4

    val requiredComponents = setOf(
        "unet",
        "vae_encoder",
        "vae_decoder",
        "text_encoder",
    )

    data class Tensor(
        val name: String,
        val dataType: String,
        val shape: List<Int>,
        val stateKey: String = name,
    )

    data class Component(
        val inputs: List<Tensor>,
        val outputs: List<Tensor>,
    )

    data class Manifest(
        val components: Map<String, Component>,
        val scheduler: DreamLiteScheduler.Config? = null,
    )

    sealed interface ParseResult {
        data class Valid(val manifest: Manifest) : ParseResult
        data class Invalid(val reason: String) : ParseResult
    }

    fun parse(file: File): ParseResult {
        if (!file.isFile || file.length() <= 0L) {
            return ParseResult.Invalid("DreamLite ABI manifest is missing or empty")
        }
        val root = runCatching { JSONObject(file.readText()) }.getOrElse {
            return ParseResult.Invalid("DreamLite ABI manifest is not valid JSON")
        }
        if (root.optInt("abi_version", -1) != VERSION) {
            return ParseResult.Invalid("unsupported DreamLite ABI version")
        }
        if (root.optString("runtime") != DreamLiteLiteRt.RUNTIME) {
            return ParseResult.Invalid("DreamLite ABI runtime mismatch")
        }
        if (root.optInt("steps", -1) != EXPECTED_STEPS) {
            return ParseResult.Invalid("DreamLite ABI must use $EXPECTED_STEPS steps")
        }
        val schedulerJson = root.optJSONObject("scheduler")
            ?: return ParseResult.Invalid("DreamLite ABI scheduler is missing")
        val scheduler = DreamLiteScheduler.Config(
            numTrainTimesteps = schedulerJson.optInt("num_train_timesteps", -1),
            useDynamicShifting = schedulerJson.optBoolean("use_dynamic_shifting", false),
            timeShiftType = schedulerJson.optString("time_shift_type"),
            baseImageSeqLen = schedulerJson.optInt("base_image_seq_len", 256),
            maxImageSeqLen = schedulerJson.optInt("max_image_seq_len", 4096),
            baseShift = schedulerJson.optDouble("base_shift", 0.5).toFloat(),
            maxShift = schedulerJson.optDouble("max_shift", 1.16).toFloat(),
        )
        if (scheduler.numTrainTimesteps <= 0 ||
            scheduler.timeShiftType !in setOf("exponential", "linear") ||
            scheduler.baseImageSeqLen <= 0 ||
            scheduler.maxImageSeqLen <= scheduler.baseImageSeqLen
        ) {
            return ParseResult.Invalid("DreamLite ABI scheduler is invalid")
        }

        val componentsJson = root.optJSONObject("components")
            ?: return ParseResult.Invalid("DreamLite ABI components are missing")
        if (requiredComponents.any { !componentsJson.has(it) }) {
            return ParseResult.Invalid("DreamLite ABI is missing required components")
        }

        fun tensors(component: JSONObject, key: String): List<Tensor>? {
            val array = component.optJSONArray(key) ?: return null
            return buildList {
                for (i in 0 until array.length()) {
                    val tensor = array.optJSONObject(i) ?: return null
                    val name = tensor.optString("name").takeIf { it.isNotBlank() } ?: return null
                    val dtype = tensor.optString("dtype").takeIf { it.isNotBlank() } ?: return null
                    val shapeJson = tensor.optJSONArray("shape") ?: return null
                    val shape = buildList {
                        for (d in 0 until shapeJson.length()) {
                            val value = shapeJson.optInt(d, Int.MIN_VALUE)
                            if (value == Int.MIN_VALUE || value == 0 || value < -1) return null
                            add(value)
                        }
                    }
                    if (shape.isEmpty()) return null
                    val stateKey = tensor.optString("state_key", name)
                        .takeIf { it.isNotBlank() } ?: return null
                    add(Tensor(name, dtype, shape, stateKey))
                }
            }
        }

        val parsed = linkedMapOf<String, Component>()
        for (name in requiredComponents) {
            val component = componentsJson.optJSONObject(name)
                ?: return ParseResult.Invalid("DreamLite ABI component $name is invalid")
            val inputs = tensors(component, "inputs")
                ?: return ParseResult.Invalid("DreamLite ABI $name inputs are invalid")
            val outputs = tensors(component, "outputs")
                ?: return ParseResult.Invalid("DreamLite ABI $name outputs are invalid")
            if (inputs.isEmpty() || outputs.isEmpty()) {
                return ParseResult.Invalid("DreamLite ABI $name tensors cannot be empty")
            }
            parsed[name] = Component(inputs, outputs)
        }
        return ParseResult.Valid(Manifest(parsed, scheduler))
    }
}
