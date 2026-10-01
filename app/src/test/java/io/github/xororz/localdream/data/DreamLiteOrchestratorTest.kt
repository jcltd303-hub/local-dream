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
