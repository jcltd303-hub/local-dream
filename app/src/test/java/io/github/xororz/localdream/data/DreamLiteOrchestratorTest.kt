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
