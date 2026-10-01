package io.github.xororz.localdream.data

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import org.junit.Rule
import org.junit.rules.TemporaryFolder

class ModelConfigIdentityTest {
    @get:Rule
    val temp = TemporaryFolder()

    @Test
    fun readsIdentityAdapterMetadata() {
        val dir = temp.newFolder("model")
        File(dir, "config.json").writeText(
            """{
              "identity_vision_encoder": "vision_encoder.bin",
              "identity_adapter": "identity_adapter.bin",
              "identity_adapter_scale": 0.85
            }"""
        )

        val config = ModelConfig.read(dir)!!
        assertEquals("vision_encoder.bin", config.identityVisionEncoder)
        assertEquals("identity_adapter.bin", config.identityAdapter)
        assertEquals(0.85f, config.identityAdapterScale)
    }

    @Test
    fun clampsIdentityAdapterScale() {
        val dir = temp.newFolder("model")
        File(dir, "config.json").writeText("""{"identity_adapter_scale": 99.0}""")
        assertEquals(2f, ModelConfig.read(dir)!!.identityAdapterScale)
    }
}
