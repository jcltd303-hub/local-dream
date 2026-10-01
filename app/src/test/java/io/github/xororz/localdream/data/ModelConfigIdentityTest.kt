package io.github.xororz.localdream.data

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
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

    @Test
    fun readsDreamLiteLiteRtMetadata() {
        val dir = temp.newFolder("dreamlite")
        File(dir, "config.json").writeText(
            """{
              "runtime": "dreamlite_litert",
              "default_steps": 4,
              "dreamlite_unet": "dreamlite_unet.tflite",
              "dreamlite_vae_encoder": "dreamlite_vae_encoder.tflite",
              "dreamlite_vae_decoder": "dreamlite_vae_decoder.tflite",
              "dreamlite_text_encoder": "dreamlite_text_encoder.tflite"
            }"""
        )

        val config = ModelConfig.read(dir)!!
        assertEquals("dreamlite_litert", config.runtime)
        assertEquals(4f, config.steps)
        assertEquals("dreamlite_unet.tflite", config.dreamliteUnet)
        assertEquals("dreamlite_vae_encoder.tflite", config.dreamliteVaeEncoder)
        assertEquals("dreamlite_vae_decoder.tflite", config.dreamliteVaeDecoder)
        assertEquals("dreamlite_text_encoder.tflite", config.dreamliteTextEncoder)
    }

    @Test
    fun rejectsUnknownRuntime() {
        val dir = temp.newFolder("unknown-runtime")
        File(dir, "config.json").writeText("""{"runtime":"not_a_runtime"}""")
        assertEquals(null, ModelConfig.read(dir)!!.runtime)
    }
}
