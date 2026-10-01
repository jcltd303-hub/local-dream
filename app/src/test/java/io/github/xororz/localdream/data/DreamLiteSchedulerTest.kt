package io.github.xororz.localdream.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class DreamLiteSchedulerTest {
    private val config = DreamLiteScheduler.Config(
        numTrainTimesteps = 1000,
        useDynamicShifting = true,
        timeShiftType = "exponential",
    )

    @Test
    fun calculatesReferenceMuFormula() {
        assertEquals(0.5f, DreamLiteScheduler.calculateMu(256, config), 1e-6f)
        assertEquals(1.16f, DreamLiteScheduler.calculateMu(4096, config), 1e-6f)
    }

    @Test
    fun scheduleAppendsTerminalZeroSigma() {
        val schedule = DreamLiteScheduler.schedule(256, config)
        assertEquals(5, schedule.size)
        assertEquals(1f, schedule.first(), 1e-6f)
        assertEquals(0f, schedule.last(), 0f)
    }

    @Test
    fun eulerStepMatchesDiffusersFormula() {
        assertArrayEquals(
            floatArrayOf(8f, 16f),
            DreamLiteScheduler.eulerStep(
                floatArrayOf(10f, 20f),
                floatArrayOf(4f, 8f),
                1f,
                0.5f,
            ),
            1e-6f,
        )
    }
}
