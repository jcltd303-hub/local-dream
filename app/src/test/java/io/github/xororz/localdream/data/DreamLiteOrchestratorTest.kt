package io.github.xororz.localdream.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DreamLiteOrchestratorTest {
    @Test
    fun usesOfficialFourStepMobileSigmaSchedule() {
        assertEquals(
            listOf(1.0f, 0.75f, 0.5f, 0.25f),
            DreamLiteOrchestrator.DEFAULT_SIGMAS,
        )
    }

    @Test
    fun generationRunsExactlyFourDenoisePasses() {
        val plan = DreamLiteOrchestrator.plan(false)
        assertFalse(plan.hasReferenceImage)
        assertEquals(
            4,
            plan.stages.count { it == DreamLiteOrchestrator.Stage.UNET },
        )
        assertEquals(DreamLiteOrchestrator.Stage.TEXT_ENCODER, plan.stages.first())
        assertEquals(DreamLiteOrchestrator.Stage.VAE_DECODER, plan.stages.last())
    }

    @Test
    fun executorReceivesStableDenoiseStepIndices() {
        val calls = mutableListOf<Pair<DreamLiteOrchestrator.Stage, Int?>>()
        DreamLiteOrchestrator.execute(
            DreamLiteOrchestrator.plan(false),
            DreamLiteOrchestrator.StageExecutor { stage, step ->
                calls += stage to step
            },
        )
        assertEquals(listOf(0, 1, 2, 3), calls.mapNotNull { it.second })
        assertEquals(null, calls.first().second)
        assertEquals(null, calls.last().second)
    }

    @Test
    fun stateCarriesOutputsAcrossStagesWithoutModelSpecificNames() {
        val state = DreamLiteOrchestrator.PipelineState()
        DreamLiteOrchestrator.executeStateful(
            DreamLiteOrchestrator.plan(false),
            state,
            DreamLiteOrchestrator.StatefulStageExecutor { stage, step, pipeline ->
                when {
                    stage == DreamLiteOrchestrator.Stage.TEXT_ENCODER ->
                        pipeline.putAll(mapOf("conditioning" to floatArrayOf(1f)))
                    stage == DreamLiteOrchestrator.Stage.UNET ->
                        pipeline.putAll(mapOf("latent" to floatArrayOf((step ?: -1).toFloat())))
                    stage == DreamLiteOrchestrator.Stage.VAE_DECODER -> {
                        assertEquals(1f, pipeline.require(listOf("conditioning")).getValue("conditioning")[0])
                        assertEquals(3f, pipeline.require(listOf("latent")).getValue("latent")[0])
                    }
                }
            },
        )
    }

    @Test
    fun componentRoutingUsesExplicitStateKeys() {
        val runtime = object : DreamLiteRuntime {
            override fun inspect() = DreamLiteRuntime.Diagnostics(
                "test", "npu", emptyList(), false
            )
            override fun runFloatComponent(
                component: String,
                inputs: Map<String, FloatArray>,
            ): Map<String, FloatArray> {
                assertEquals(7f, inputs.getValue("model_input")[0])
                return mapOf("model_output" to floatArrayOf(9f))
            }
        }
        val manifest = DreamLiteAbi.Manifest(
            mapOf(
                "unet" to DreamLiteAbi.Component(
                    listOf(DreamLiteAbi.Tensor("model_input", "float32", listOf(1), "latent")),
                    listOf(DreamLiteAbi.Tensor("model_output", "float32", listOf(1), "next_latent")),
                )
            )
        )
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf("latent" to floatArrayOf(7f))
        )
        DreamLiteOrchestrator.runComponent(runtime, manifest, "unet", state)
        assertEquals(9f, state.tensors.getValue("next_latent")[0])
    }

    @Test
    fun runtimeExecutionMapsStagesAndRunsFourUnetPasses() {
        val calls = mutableListOf<String>()
        val runtime = object : DreamLiteRuntime {
            override fun inspect() = DreamLiteRuntime.Diagnostics(
                "test", "npu", emptyList(), false
            )
            override fun runFloatComponent(
                component: String,
                inputs: Map<String, FloatArray>,
            ): Map<String, FloatArray> {
                calls += component
                val input = inputs.values.first()[0]
                return mapOf(component + "_out" to floatArrayOf(input + 1f))
            }
        }
        fun component(name: String, inputKey: String, outputKey: String) =
            DreamLiteAbi.Component(
                listOf(DreamLiteAbi.Tensor(name + "_in", "float32", listOf(1), inputKey)),
                listOf(DreamLiteAbi.Tensor(name + "_out", "float32", listOf(1), outputKey)),
            )
        val manifest = DreamLiteAbi.Manifest(
            mapOf(
                "text_encoder" to component("text_encoder", "prompt", "conditioning"),
                "unet" to component("unet", "latent", "latent"),
                "vae_decoder" to component("vae_decoder", "latent", "image"),
            )
        )
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf("prompt" to floatArrayOf(1f), "latent" to floatArrayOf(0f))
        )
        val steps = mutableListOf<Int>()
        DreamLiteOrchestrator.executeRuntime(
            DreamLiteOrchestrator.plan(false),
            runtime,
            manifest,
            state,
        ) { step, _ -> steps += step }
        assertEquals(listOf(0, 1, 2, 3), steps)
        assertEquals(
            listOf("text_encoder", "unet", "unet", "unet", "unet", "vae_decoder"),
            calls,
        )
        assertEquals(5f, state.tensors.getValue("image")[0])
    }

    @Test
    fun scheduledStepUpdatesLatentAfterModelOutput() {
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf(
                DreamLiteOrchestrator.LATENT_STATE_KEY to floatArrayOf(10f),
                DreamLiteOrchestrator.MODEL_OUTPUT_STATE_KEY to floatArrayOf(4f),
            )
        )
        DreamLiteOrchestrator.executeScheduledStep(state, 1f, 0.5f)
        assertEquals(
            8f,
            state.tensors.getValue(DreamLiteOrchestrator.LATENT_STATE_KEY)[0],
            1e-6f,
        )
    }

    @Test
    fun scheduledRuntimeInjectsTimestepsAndUpdatesLatentFourTimes() {
        val seenTimesteps = mutableListOf<Float>()
        val runtime = object : DreamLiteRuntime {
            override fun inspect() = DreamLiteRuntime.Diagnostics("test", "npu", emptyList(), false)
            override fun runFloatComponent(
                component: String,
                inputs: Map<String, FloatArray>,
            ): Map<String, FloatArray> = when (component) {
                "text_encoder" -> mapOf("conditioning_out" to floatArrayOf(1f))
                "unet" -> {
                    seenTimesteps += inputs.getValue("timestep")[0]
                    mapOf("noise" to floatArrayOf(1f))
                }
                "vae_decoder" -> mapOf("image_out" to inputs.getValue("latent"))
                else -> error("unexpected component")
            }
        }
        val manifest = DreamLiteAbi.Manifest(
            components = mapOf(
                "text_encoder" to DreamLiteAbi.Component(
                    listOf(DreamLiteAbi.Tensor("prompt", "float32", listOf(1), "prompt")),
                    listOf(DreamLiteAbi.Tensor("conditioning_out", "float32", listOf(1), "conditioning")),
                ),
                "unet" to DreamLiteAbi.Component(
                    listOf(
                        DreamLiteAbi.Tensor("latent", "float32", listOf(1), "latent"),
                        DreamLiteAbi.Tensor("timestep", "float32", listOf(1), "timestep"),
                    ),
                    listOf(DreamLiteAbi.Tensor("noise", "float32", listOf(1), "model_output")),
                ),
                "vae_decoder" to DreamLiteAbi.Component(
                    listOf(DreamLiteAbi.Tensor("latent", "float32", listOf(1), "latent")),
                    listOf(DreamLiteAbi.Tensor("image_out", "float32", listOf(1), "image")),
                ),
            ),
            scheduler = DreamLiteScheduler.Config(
                numTrainTimesteps = 1000,
                useDynamicShifting = false,
                timeShiftType = "exponential",
            ),
        )
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf("prompt" to floatArrayOf(1f), "latent" to floatArrayOf(10f))
        )
        DreamLiteOrchestrator.executeScheduledRuntime(
            DreamLiteOrchestrator.plan(false),
            runtime,
            manifest,
            state,
            imageSeqLen = 256,
        )
        assertEquals(listOf(1000f, 750f, 500f, 250f), seenTimesteps)
        assertEquals(9f, state.tensors.getValue("image")[0], 1e-6f)
    }

    @Test
    fun scheduledRuntimeBuildsAndCropsSpatialConditioningEachStep() {
        val modelInputs = mutableListOf<FloatArray>()
        val runtime = object : DreamLiteRuntime {
            override fun inspect() = DreamLiteRuntime.Diagnostics("test", "npu", emptyList(), false)
            override fun runFloatComponent(component: String, inputs: Map<String, FloatArray>) =
                when (component) {
                    "text_encoder" -> mapOf("conditioning_out" to floatArrayOf(1f))
                    "unet" -> {
                        modelInputs += inputs.getValue("sample").copyOf()
                        mapOf("noise" to floatArrayOf(1f, 1f, 99f, 99f))
                    }
                    "vae_decoder" -> mapOf("image_out" to inputs.getValue("latent"))
                    else -> error("unexpected component")
                }
        }
        val manifest = DreamLiteAbi.Manifest(
            components = mapOf(
                "text_encoder" to DreamLiteAbi.Component(
                    listOf(DreamLiteAbi.Tensor("prompt", "float32", listOf(1), "prompt")),
                    listOf(DreamLiteAbi.Tensor("conditioning_out", "float32", listOf(1), "conditioning")),
                ),
                "unet" to DreamLiteAbi.Component(
                    listOf(
                        DreamLiteAbi.Tensor("sample", "float32", listOf(1,1,1,4), "model_input"),
                        DreamLiteAbi.Tensor("timestep", "float32", listOf(1), "timestep"),
                        DreamLiteAbi.Tensor("conditioning", "float32", listOf(1), "conditioning"),
                        DreamLiteAbi.Tensor("attention", "float32", listOf(1), "attention_mask"),
                        DreamLiteAbi.Tensor("time_ids", "float32", listOf(1,2), "time_ids"),
                    ),
                    listOf(DreamLiteAbi.Tensor("noise", "float32", listOf(1,1,1,4), "model_output")),
                ),
                "vae_decoder" to DreamLiteAbi.Component(
                    listOf(DreamLiteAbi.Tensor("latent", "float32", listOf(1,1,1,2), "latent")),
                    listOf(DreamLiteAbi.Tensor("image_out", "float32", listOf(1), "image")),
                ),
            ),
            scheduler = DreamLiteScheduler.Config(1000, false, "exponential"),
        )
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf(
                "prompt" to floatArrayOf(1f),
                "latent" to floatArrayOf(10f, 20f),
                "reference_latent" to floatArrayOf(30f, 40f),
                "attention_mask" to floatArrayOf(1f),
            )
        )
        DreamLiteOrchestrator.executeScheduledRuntime(
            DreamLiteOrchestrator.plan(false),
            runtime,
            manifest,
            state,
            imageSeqLen = 1,
            latentShape = DreamLiteOrchestrator.LatentShape(1, 1, 1, 2),
            outputWidth = 16,
            outputHeight = 8,
        )
        assertEquals(4, modelInputs.size)
        assertArrayEquals(floatArrayOf(10f, 20f, 30f, 40f), modelInputs.first(), 0f)
        assertArrayEquals(floatArrayOf(9f, 19f), state.tensors.getValue("image"), 1e-6f)
        assertArrayEquals(floatArrayOf(16f, 8f), state.tensors.getValue("time_ids"), 0f)
    }


    @Test
    fun externalConditionerBypassesTextGraphAndFeedsUnet() {
        val calls = mutableListOf<String>()
        val runtime = object : DreamLiteRuntime {
            override fun inspect() = DreamLiteRuntime.Diagnostics("test", "npu", emptyList(), false)
            override fun runFloatComponent(component: String, inputs: Map<String, FloatArray>) =
                when (component) {
                    "text_encoder" -> error("external conditioner must bypass text graph")
                    "unet" -> {
                        calls += component
                        assertArrayEquals(floatArrayOf(7f), inputs.getValue("conditioning"), 0f)
                        assertArrayEquals(floatArrayOf(1f), inputs.getValue("mask"), 0f)
                        mapOf("noise" to floatArrayOf(0f))
                    }
                    "vae_decoder" -> mapOf("image" to inputs.getValue("latent"))
                    else -> error("unexpected component")
                }
        }
        val manifest = DreamLiteAbi.Manifest(
            components = mapOf(
                "text_encoder" to DreamLiteAbi.Component(emptyList(), emptyList()),
                "unet" to DreamLiteAbi.Component(
                    listOf(
                        DreamLiteAbi.Tensor("latent", "float32", listOf(1), "latent"),
                        DreamLiteAbi.Tensor("timestep", "float32", listOf(1), "timestep"),
                        DreamLiteAbi.Tensor("conditioning", "float32", listOf(1), "conditioning"),
                        DreamLiteAbi.Tensor("mask", "float32", listOf(1), "attention_mask"),
                    ),
                    listOf(DreamLiteAbi.Tensor("noise", "float32", listOf(1), "model_output")),
                ),
                "vae_decoder" to DreamLiteAbi.Component(
                    listOf(DreamLiteAbi.Tensor("latent", "float32", listOf(1), "latent")),
                    listOf(DreamLiteAbi.Tensor("image", "float32", listOf(1), "image")),
                ),
            ),
            scheduler = DreamLiteScheduler.Config(1000, false, "exponential"),
        )
        val conditioner = object : DreamLiteConditioner {
            override fun encode(request: DreamLiteConditioning.Request) =
                DreamLiteConditioner.Output(floatArrayOf(7f), floatArrayOf(1f), 1, 2048)
        }
        val state = DreamLiteOrchestrator.PipelineState(
            linkedMapOf("latent" to floatArrayOf(2f))
        )
        DreamLiteOrchestrator.executeScheduledRuntime(
            DreamLiteOrchestrator.plan(false), runtime, manifest, state, 256,
            conditioner = conditioner,
            conditioningRequest = DreamLiteConditioning.Request(
                DreamLiteConditioning.Mode.GENERATE, "portrait"
            ),
        )
        assertEquals(4, calls.size)
    }

    @Test
    fun editEncodesReferenceBeforeDenoising() {
        val plan = DreamLiteOrchestrator.plan(true)
        assertTrue(plan.hasReferenceImage)
        assertEquals(
            listOf(
                DreamLiteOrchestrator.Stage.TEXT_ENCODER,
                DreamLiteOrchestrator.Stage.REFERENCE_ENCODER,
            ),
            plan.stages.take(2),
        )
        assertEquals(
            4,
            plan.stages.count { it == DreamLiteOrchestrator.Stage.UNET },
        )
    }
}
