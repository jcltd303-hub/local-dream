package io.github.xororz.localdream.data

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class DreamLiteTensorOpsTest {
    @Test
    fun timeIdsUseWidthThenHeight() {
        assertArrayEquals(floatArrayOf(1024f, 768f), DreamLiteTensorOps.timeIds(1024, 768), 0f)
    }

    @Test
    fun concatenatesReferenceLatentAlongWidth() {
        val out = DreamLiteTensorOps.concatWidth(
            floatArrayOf(1f, 2f, 3f, 4f),
            floatArrayOf(5f, 6f, 7f, 8f),
            batch = 1, channels = 1, height = 2, width = 2,
        )
        assertArrayEquals(
            floatArrayOf(1f, 2f, 5f, 6f, 3f, 4f, 7f, 8f),
            out,
            0f,
        )
    }

    @Test
    fun cropsUnetPredictionBackToLatentWidth() {
        val out = DreamLiteTensorOps.cropNoiseToLatentWidth(
            floatArrayOf(1f, 2f, 9f, 9f, 3f, 4f, 9f, 9f),
            batch = 1, channels = 1, height = 2, modelWidth = 4, latentWidth = 2,
        )
        assertArrayEquals(floatArrayOf(1f, 2f, 3f, 4f), out, 0f)
    }

    @Test
    fun imageSequenceLengthMatchesReferencePipeline() {
        assertEquals(4096, DreamLiteTensorOps.imageSequenceLength(128, 128))
    }
}
