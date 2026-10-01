package io.github.xororz.localdream.data

/**
 * Manifest-driven DreamLite Mobile execution order.
 *
 * Tensor names and shapes remain owned by dreamlite_abi.json. This layer only
 * freezes the distilled mobile schedule: conditioning, optional reference
 * encoding, exactly four denoise passes, then decode.
 */
object DreamLiteOrchestrator {
    const val DENOISE_STEPS = DreamLiteAbi.EXPECTED_STEPS

    enum class Stage { TEXT_ENCODER, REFERENCE_ENCODER, UNET, VAE_DECODER }

    data class Plan(
        val hasReferenceImage: Boolean,
        val stages: List<Stage>,
    )

    /**
     * Runs a frozen execution plan through a caller-owned stage executor.
     *
     * The executor owns tensor routing because only dreamlite_abi.json may
     * define converted graph tensor names/shapes. This keeps scheduling
     * testable without inventing an Android-side model ABI.
     */
    fun interface StageExecutor {
        fun run(stage: Stage, denoiseStep: Int?)
    }

    fun execute(plan: Plan, executor: StageExecutor) {
        var denoiseStep = 0
        plan.stages.forEach { stage ->
            if (stage == Stage.UNET) {
                executor.run(stage, denoiseStep++)
            } else {
                executor.run(stage, null)
            }
        }
        check(denoiseStep == DENOISE_STEPS) {
            "DreamLite execution must run exactly $DENOISE_STEPS denoise steps"
        }
    }

    fun plan(hasReferenceImage: Boolean): Plan {
        val stages = buildList {
            add(Stage.TEXT_ENCODER)
            if (hasReferenceImage) add(Stage.REFERENCE_ENCODER)
            repeat(DENOISE_STEPS) { add(Stage.UNET) }
            add(Stage.VAE_DECODER)
        }
        return Plan(hasReferenceImage, stages)
    }
}
