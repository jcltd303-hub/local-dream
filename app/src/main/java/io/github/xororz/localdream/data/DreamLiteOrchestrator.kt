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

    const val LATENT_STATE_KEY = "latent"
    const val MODEL_INPUT_STATE_KEY = "model_input"
    const val MODEL_OUTPUT_STATE_KEY = "model_output"
    const val TIMESTEP_STATE_KEY = "timestep"
    const val CONDITIONING_STATE_KEY = "conditioning"
    const val ATTENTION_MASK_STATE_KEY = "attention_mask"
    const val TIME_IDS_STATE_KEY = "time_ids"
    const val REFERENCE_LATENT_STATE_KEY = "reference_latent"

    data class LatentShape(
        val batch: Int = 1,
        val channels: Int = 4,
        val height: Int,
        val width: Int,
    )

    fun prepareSpatialConditioning(
        state: PipelineState,
        shape: LatentShape,
        outputWidth: Int,
        outputHeight: Int,
    ) {
        val latent = state.tensors[LATENT_STATE_KEY]
            ?: error("DreamLite pipeline tensor latent is unavailable")
        val reference = state.tensors[REFERENCE_LATENT_STATE_KEY]
            ?: FloatArray(latent.size)
        state.tensors[MODEL_INPUT_STATE_KEY] = DreamLiteTensorOps.concatWidth(
            latent,
            reference,
            shape.batch,
            shape.channels,
            shape.height,
            shape.width,
        )
        state.tensors[TIME_IDS_STATE_KEY] = DreamLiteTensorOps.timeIds(outputWidth, outputHeight)
    }

    fun cropModelOutputToLatent(
        state: PipelineState,
        shape: LatentShape,
    ) {
        val modelOutput = state.tensors[MODEL_OUTPUT_STATE_KEY]
            ?: error("DreamLite pipeline tensor model_output is unavailable")
        state.tensors[MODEL_OUTPUT_STATE_KEY] = DreamLiteTensorOps.cropNoiseToLatentWidth(
            modelOutput,
            shape.batch,
            shape.channels,
            shape.height,
            shape.width * 2,
            shape.width,
        )
    }

    fun executeScheduledRuntime(
        plan: Plan,
        runtime: DreamLiteRuntime,
        manifest: DreamLiteAbi.Manifest,
        state: PipelineState,
        imageSeqLen: Int,
    ): PipelineState {
        val scheduler = requireNotNull(manifest.scheduler) {
            "DreamLite scheduler config is required"
        }
        val sigmas = DreamLiteScheduler.schedule(imageSeqLen, scheduler)
        return executeRuntime(
            plan,
            runtime,
            manifest,
            state,
            beforeDenoise = { step, pipeline ->
                val sigma = sigmas[step]
                pipeline.tensors[TIMESTEP_STATE_KEY] =
                    floatArrayOf(DreamLiteScheduler.timestep(sigma, scheduler))
            },
            afterDenoise = { step, pipeline ->
                executeScheduledStep(pipeline, sigmas[step], sigmas[step + 1])
            },
        )
    }

    fun executeScheduledStep(
        state: PipelineState,
        sigma: Float,
        sigmaNext: Float,
    ) {
        val latent = state.tensors[LATENT_STATE_KEY]
            ?: error("DreamLite pipeline tensor latent is unavailable")
        val modelOutput = state.tensors[MODEL_OUTPUT_STATE_KEY]
            ?: error("DreamLite pipeline tensor model_output is unavailable")
        state.tensors[LATENT_STATE_KEY] =
            DreamLiteScheduler.eulerStep(latent, modelOutput, sigma, sigmaNext)
    }

    fun executeRuntime(
        plan: Plan,
        runtime: DreamLiteRuntime,
        manifest: DreamLiteAbi.Manifest,
        state: PipelineState,
        beforeDenoise: (Int, PipelineState) -> Unit = { _, _ -> },
        afterDenoise: (Int, PipelineState) -> Unit = { _, _ -> },
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
                    afterDenoise(index, pipeline)
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
