package io.github.xororz.localdream.data

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class DreamLiteLiteRtTest {
    @get:Rule
    val temp = TemporaryFolder()

    private fun writeAbi(dir: File, version: Int = 1, steps: Int = 4) {
        val generic = """{"inputs":[{"name":"in","dtype":"float32","shape":[1,1]}],"outputs":[{"name":"out","dtype":"float32","shape":[1,1]}]}"""
        val unet = """{"inputs":[{"name":"sample","dtype":"float32","shape":[1,4,128,256],"state_key":"model_input"},{"name":"timestep","dtype":"float32","shape":[1],"state_key":"timestep"},{"name":"encoder_hidden_states","dtype":"float32","shape":[1,77,2048],"state_key":"conditioning"},{"name":"encoder_attention_mask","dtype":"float32","shape":[1,77],"state_key":"attention_mask"},{"name":"time_ids","dtype":"float32","shape":[1,2],"state_key":"time_ids"}],"outputs":[{"name":"noise_pred","dtype":"float32","shape":[1,4,128,256],"state_key":"model_output"}]}"""
        val decoder = """{"inputs":[{"name":"latent","dtype":"float32","shape":[1,1],"state_key":"latent"}],"outputs":[{"name":"image","dtype":"float32","shape":[1,1],"state_key":"image"}]}"""
        File(dir, DreamLiteAbi.MANIFEST).writeText(
            """{"abi_version":$version,"runtime":"dreamlite_litert","steps":$steps,"scheduler":{"num_train_timesteps":1000,"use_dynamic_shifting":true,"time_shift_type":"exponential","base_image_seq_len":256,"max_image_seq_len":4096,"base_shift":0.5,"max_shift":1.16},"vae":{"scaling_factor":1.0,"shift_factor":0.0},"components":{"unet":$unet,"vae_encoder":$generic,"vae_decoder":$decoder,"text_encoder":$generic}}""",
        )
    }

    private fun config(unet: String = "unet.tflite") = ModelConfig(
        runtime = DreamLiteLiteRt.RUNTIME,
        dreamliteUnet = unet,
        dreamliteVaeEncoder = "ve.tflite",
        dreamliteVaeDecoder = "vd.tflite",
        dreamliteTextEncoder = "te.tflite",
    )

    @Test
    fun acceptsCompleteNonEmptyPackage() {
        val dir = temp.newFolder("dreamlite")
        listOf("unet.tflite", "ve.tflite", "vd.tflite", "te.tflite").forEach {
            File(dir, it).writeBytes(byteArrayOf(1))
        }
        writeAbi(dir)
        assertTrue(DreamLiteLiteRt.probe(dir, config()) is DreamLiteLiteRt.ProbeResult.Ready)
    }

    @Test
    fun rejectsWrongAbiVersion() {
        val dir = temp.newFolder("wrong-abi")
        listOf("unet.tflite", "ve.tflite", "vd.tflite", "te.tflite").forEach {
            File(dir, it).writeBytes(byteArrayOf(1))
        }
        writeAbi(dir, version = 2)
        assertTrue(DreamLiteLiteRt.probe(dir, config()) is DreamLiteLiteRt.ProbeResult.Invalid)
    }

    @Test
    fun rejectsWrongStepCount() {
        val dir = temp.newFolder("wrong-steps")
        listOf("unet.tflite", "ve.tflite", "vd.tflite", "te.tflite").forEach {
            File(dir, it).writeBytes(byteArrayOf(1))
        }
        writeAbi(dir, steps = 8)
        assertTrue(DreamLiteLiteRt.probe(dir, config()) is DreamLiteLiteRt.ProbeResult.Invalid)
    }

    @Test
    fun rejectsEmptyComponent() {
        val dir = temp.newFolder("empty")
        File(dir, "unet.tflite").writeBytes(byteArrayOf())
        listOf("ve.tflite", "vd.tflite", "te.tflite").forEach {
            File(dir, it).writeBytes(byteArrayOf(1))
        }
        writeAbi(dir)
        assertTrue(DreamLiteLiteRt.probe(dir, config()) is DreamLiteLiteRt.ProbeResult.Invalid)
    }

    @Test
    fun rejectsPathTraversal() {
        val root = temp.newFolder("root")
        temp.newFile("outside.tflite").writeBytes(byteArrayOf(1))
        listOf("ve.tflite", "vd.tflite", "te.tflite").forEach {
            File(root, it).writeBytes(byteArrayOf(1))
        }
        writeAbi(root)
        assertTrue(
            DreamLiteLiteRt.probe(root, config("../outside.tflite")) is
                DreamLiteLiteRt.ProbeResult.Invalid,
        )
    }
}
