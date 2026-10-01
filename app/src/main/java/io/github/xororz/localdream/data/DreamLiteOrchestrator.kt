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

    /** Official DreamLite Mobile default: linspace(1.0, 1 / steps, steps). */
    val DEFAULT_SIGMAS: List<Float> =
        List(DENOISE_STEPS) { index ->
            1.0f - index * ((1.0f - 1.0f / DENOISE_STEPS) / (DENOISE_STEPS - 1))
        }


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
    data class PipelineState(
        val tensors: MutableMap<String, FloatArray> = linkedMapOf(),
    ) {
        fun putAll(outputs: Map<String, FloatArray>) {
            tensors.putAll(outputs)
        }

        fun require(names: Collection<String>): Map<String, FloatArray> =
            names.associateWith { name ->
                tensors[name] ?: error("DreamLite pipeline tensor $name is unavailable")
            }
    }

    fun interface StageExecutor {
        fun run(stage: Stage, denoiseStep: Int?)
    }

    fun interface StatefulStageExecutor {
        fun run(stage: Stage, denoiseStep: Int?, state: PipelineState)
    }

    fun execute(plan: Plan, executor: StageExecutor) =
        executeStateful(plan, PipelineState()) { stage, step, _ ->
            executor.run(stage, step)
        }

    fun runComponent(
        runtime: DreamLiteRuntime,
        manifest: DreamLiteAbi.Manifest,
        component: String,
        state: PipelineState,
    ) {
        val abi = manifest.components[component]
            ?: error("DreamLite ABI has no component " + component)
        val inputs = abi.inputs.associate { tensor ->
            tensor.name to (
                state.tensors[tensor.stateKey]
                    ?: error("DreamLite pipeline tensor " + tensor.stateKey + " is unavailable")
                )
        }
        val outputs = runtime.runFloatComponent(component, inputs)
        abi.outputs.forEach { tensor ->
            val value = outputs[tensor.name]
                ?: error("DreamLite component " + component + " did not return " + tensor.name)
            state.tensors[tensor.stateKey] = value
        }
    }

    fun executeRuntime(
        plan: Plan,
        runtime: DreamLiteRuntime,
        manifest: DreamLiteAbi.Manifest,
        state: PipelineState,
        beforeDenoise: (Int, PipelineState) -> Unit = { _, _ -> },
    ): PipelineState =
        executeStateful(plan, state) { stage, step, pipeline ->
            when (stage) {
                Stage.TEXT_ENCODER ->
                    runComponent(runtime, manifest, "text_encoder", pipeline)
                Stage.REFERENCE_ENCODER ->
                    runComponent(runtime, manifest, "vae_encoder", pipeline)
                Stage.UNET -> {
                    val index = requireNotNull(step)
                    beforeDenoise(index, pipeline)
                    runComponent(runtime, manifest, "unet", pipeline)
                }
                Stage.VAE_DECODER ->
                    runComponent(runtime, manifest, "vae_decoder", pipeline)
            }
        }

    fun executeStateful(
        plan: Plan,
        state: PipelineState,
        executor: StatefulStageExecutor,
    ): PipelineState {
        var denoiseStep = 0
        plan.stages.forEach { stage ->
            if (stage == Stage.UNET) {
                executor.run(stage, denoiseStep++, state)
            } else {
                executor.run(stage, null, state)
            }
        }
        check(denoiseStep == DENOISE_STEPS) {
            "DreamLite execution must run exactly $DENOISE_STEPS denoise steps"
        }
        return state
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
