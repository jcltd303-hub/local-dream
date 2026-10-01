package io.github.xororz.localdream.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DreamLiteOrchestratorTest {
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
