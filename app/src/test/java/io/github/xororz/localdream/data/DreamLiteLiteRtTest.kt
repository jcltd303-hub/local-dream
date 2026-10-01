package io.github.xororz.localdream.data

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class DreamLiteLiteRtTest {
    @get:Rule
    val temp = TemporaryFolder()

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
        assertTrue(DreamLiteLiteRt.probe(dir, config()) is DreamLiteLiteRt.ProbeResult.Ready)
    }

    @Test
    fun rejectsEmptyComponent() {
        val dir = temp.newFolder("empty")
        File(dir, "unet.tflite").writeBytes(byteArrayOf())
        listOf("ve.tflite", "vd.tflite", "te.tflite").forEach {
            File(dir, it).writeBytes(byteArrayOf(1))
        }
        assertTrue(DreamLiteLiteRt.probe(dir, config()) is DreamLiteLiteRt.ProbeResult.Invalid)
    }

    @Test
    fun rejectsPathTraversal() {
        val root = temp.newFolder("root")
        temp.newFile("outside.tflite").writeBytes(byteArrayOf(1))
        listOf("ve.tflite", "vd.tflite", "te.tflite").forEach {
            File(root, it).writeBytes(byteArrayOf(1))
        }
        assertTrue(
            DreamLiteLiteRt.probe(root, config("../outside.tflite")) is
                DreamLiteLiteRt.ProbeResult.Invalid,
        )
    }
}
