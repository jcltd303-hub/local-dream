package io.github.xororz.localdream.data

import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.cos
import kotlin.math.PI
import kotlin.random.Random

/** Pure orchestration entry point used by the Android service and host tests. */
object DreamLiteGeneration {
    data class Request(
        val prompt: String,
        val width: Int = 1024,
        val height: Int = 1024,
        val seed: Long,
        val referenceRgb: ByteArray? = null,
        val referenceWidth: Int = 0,
        val referenceHeight: Int = 0,
    )

    internal fun gaussianNoise(size: Int, seed: Long): FloatArray {
        require(size >= 0)
        val random = Random(seed)
        val out = FloatArray(size)
        var index = 0
        while (index < size) {
            // Box-Muller transform: DreamLite's reference pipeline initializes
            // latents with randn_tensor, so the Android path must also be N(0,1)
            // rather than uniform noise. Kotlin's RNG is intentionally local;
            // seed determinism is stable inside this backend, not bit-identical
            // to PyTorch's generator.
            val u1 = random.nextDouble().coerceAtLeast(Double.MIN_VALUE)
            val u2 = random.nextDouble()
            val radius = sqrt(-2.0 * ln(u1))
            val angle = 2.0 * PI * u2
            out[index++] = (radius * cos(angle)).toFloat()
            if (index < size) {
                out[index++] = (radius * kotlin.math.sin(angle)).toFloat()
            }
        }
        return out
    }

    fun run(
        runtime: DreamLiteRuntime,
        manifest: DreamLiteAbi.Manifest,
        request: Request,
        conditioner: DreamLiteConditioner? = null,
    ): FloatArray {
        require(request.width == 1024 && request.height == 1024) {
            "DreamLite Mobile v1 is frozen to the official 1024x1024 graph contract"
        }
        val shape = DreamLiteOrchestrator.LatentShape(
            height = request.height / 8,
            width = request.width / 8,
        )
        val latent = gaussianNoise(
            shape.batch * shape.channels * shape.height * shape.width,
            request.seed,
        )
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf(DreamLiteOrchestrator.LATENT_STATE_KEY to latent)
        )
        val hasReference = request.referenceRgb != null
        if (hasReference) {
            require(request.referenceWidth == 1024 && request.referenceHeight == 1024) {
                "DreamLite Mobile v1 edit reference must be normalized to 1024x1024"
            }
            require(request.referenceRgb!!.size == 1024 * 1024 * 3) {
                "DreamLite edit reference must be packed RGB8"
            }
            state.tensors[DreamLiteOrchestrator.REFERENCE_IMAGE_STATE_KEY] =
                request.referenceRgb!!.map { (it.toInt() and 0xff) / 127.5f - 1f }.toFloatArray()
        }
        require(conditioner != null) {
            "DreamLite generation requires the Qwen3-VL conditioner; text-only fallback is not reference-equivalent"
        }
        val conditioningRequest = DreamLiteConditioning.Request(
            if (hasReference) DreamLiteConditioning.Mode.EDIT else DreamLiteConditioning.Mode.GENERATE,
            request.prompt,
            request.referenceRgb,
            request.referenceWidth,
            request.referenceHeight,
        )
        DreamLiteOrchestrator.executeScheduledRuntime(
            DreamLiteOrchestrator.plan(hasReference),
            runtime,
            manifest,
            state,
            imageSeqLen = shape.height * shape.width / 4,
            latentShape = shape,
            outputWidth = request.width,
            outputHeight = request.height,
            conditioner = conditioner,
            conditioningRequest = conditioningRequest,
        )
        return state.tensors["image"]
            ?: error("DreamLite VAE decoder did not produce image")
    }
}
