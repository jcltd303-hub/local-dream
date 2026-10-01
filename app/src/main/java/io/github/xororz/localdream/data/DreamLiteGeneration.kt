package io.github.xororz.localdream.data

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

    fun run(
        runtime: DreamLiteRuntime,
        manifest: DreamLiteAbi.Manifest,
        request: Request,
        conditioner: DreamLiteConditioner? = null,
    ): FloatArray {
        require(request.width > 0 && request.height > 0)
        require(request.width % 8 == 0 && request.height % 8 == 0)
        val shape = DreamLiteOrchestrator.LatentShape(
            height = request.height / 8,
            width = request.width / 8,
        )
        val random = Random(request.seed)
        val latent = FloatArray(shape.batch * shape.channels * shape.height * shape.width) {
            // Deterministic centered noise for the runtime boundary. Device parity
            // tests compare this seed path against the converted reference.
            random.nextFloat() * 2f - 1f
        }
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf(DreamLiteOrchestrator.LATENT_STATE_KEY to latent)
        )
        val hasReference = request.referenceRgb != null
        if (hasReference) {
            state.tensors[DreamLiteOrchestrator.REFERENCE_IMAGE_STATE_KEY] =
                request.referenceRgb!!.map { (it.toInt() and 0xff) / 127.5f - 1f }.toFloatArray()
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
            conditioningRequest = conditioningRequest.takeIf { conditioner != null },
        )
        return state.tensors["image"]
            ?: error("DreamLite VAE decoder did not produce image")
    }
}
