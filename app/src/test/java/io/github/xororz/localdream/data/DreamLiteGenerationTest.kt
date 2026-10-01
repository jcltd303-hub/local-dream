package io.github.xororz.localdream.data

import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DreamLiteGenerationTest {
    @Test
    fun latentNoiseIsDeterministicStandardGaussian() {
        val a = DreamLiteGeneration.gaussianNoise(100_000, 42L)
        val b = DreamLiteGeneration.gaussianNoise(100_000, 42L)
        assertArrayEquals(a, b, 0f)
        val mean = a.sum().toDouble() / a.size
        val variance = a.fold(0.0) { acc, value ->
            val d = value - mean
            acc + d * d
        } / a.size
        assertTrue(kotlin.math.abs(mean) < 0.02)
        assertTrue(kotlin.math.abs(variance - 1.0) < 0.03)
    }

    @Test
    fun generationRunsConditionerFourDenoiseStepsAndDecoder() {
        val calls = mutableListOf<String>()
        val runtime = object : DreamLiteRuntime {
            override fun inspect() = DreamLiteRuntime.Diagnostics("test", "npu", emptyList(), false)
            override fun runFloatComponent(component: String, inputs: Map<String, FloatArray>): Map<String, FloatArray> {
                calls += component
                return when (component) {
                    "unet" -> mapOf("noise" to FloatArray(inputs.getValue("sample").size))
                    "vae_decoder" -> mapOf("image" to FloatArray(3 * 1024 * 1024))
                    else -> error("unexpected $component")
                }
            }
            override fun close() = Unit
        }
        fun t(name: String, state: String) = DreamLiteAbi.Tensor(name, "float32", listOf(1), state)
        val manifest = DreamLiteAbi.Manifest(
            components = mapOf(
                "unet" to DreamLiteAbi.Component(
                    listOf(
                        DreamLiteAbi.Tensor("sample","float32",listOf(1,4,128,256),"model_input"),
                        t("time","timestep"),
                        DreamLiteAbi.Tensor("cond","float32",listOf(1,512,2048),"conditioning"),
                        DreamLiteAbi.Tensor("mask","float32",listOf(1,512),"attention_mask"),
                        DreamLiteAbi.Tensor("ids","float32",listOf(1,2),"time_ids")
                    ),
                    listOf(DreamLiteAbi.Tensor("noise","float32",listOf(1,4,128,256),"model_output")),
                ),
                "vae_decoder" to DreamLiteAbi.Component(
                    listOf(DreamLiteAbi.Tensor("latent","float32",listOf(1,4,128,128),"latent")),
                    listOf(DreamLiteAbi.Tensor("image","float32",listOf(1,3,1024,1024),"image"))
                ),
            ),
            scheduler = DreamLiteScheduler.Config(1000, true, "exponential"),
            vae = DreamLiteAbi.Vae(1f, 0f),
            conditioningBackend = DreamLiteAbi.CONDITIONING_QWEN3_VL_GGUF,
        )
        val conditioner = object : DreamLiteConditioner {
            override fun encode(request: DreamLiteConditioning.Request) =
                DreamLiteConditioner.Output(FloatArray(2048), floatArrayOf(1f), 1)
            override fun close() = Unit
        }
        val image = DreamLiteGeneration.run(
            runtime, manifest, DreamLiteGeneration.Request("portrait", 1024, 1024, 7), conditioner
        )
        assertEquals(listOf("unet","unet","unet","unet","vae_decoder"), calls)
        assertEquals(3 * 1024 * 1024, image.size)
    }
}
