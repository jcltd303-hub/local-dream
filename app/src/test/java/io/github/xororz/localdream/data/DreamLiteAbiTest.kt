package io.github.xororz.localdream.data

import org.junit.Assert.assertEquals
import org.junit.Test

class DreamLiteAbiTest {
    @Test
    fun pinsFirstConvertedPackageContract() {
        assertEquals(1, DreamLiteAbi.VERSION)
        assertEquals(4, DreamLiteAbi.EXPECTED_STEPS)
        assertEquals("dreamlite_abi.json", DreamLiteAbi.MANIFEST)
        assertEquals(
            setOf("unet", "vae_encoder", "vae_decoder", "text_encoder"),
            DreamLiteAbi.requiredComponents,
        )
    }
}
