package io.github.xororz.localdream.data

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test

class DreamLiteGenerationTest {
    @Test
    fun generationRunsConditionerFourDenoiseStepsAndDecoder() {
        val calls = mutableListOf<String>()
        val runtime = object : DreamLiteRuntime {
            override fun inspect() = DreamLiteRuntime.Diagnostics("test", "npu", emptyList(), false)
            override fun runFloatComponent(component: String, inputs: Map<String, FloatArray>): Map<String, FloatArray> {
                calls += component
                return when (component) {
                    "unet" -> mapOf("noise" to FloatArray(inputs.getValue("sample").size))
                    "vae_decoder" -> mapOf("image" to inputs.getValue("latent"))
                    else -> error("unexpected $component")
                }
            }
            override fun close() = Unit
        }
        fun t(name: String, state: String) = DreamLiteAbi.Tensor(name, "float32", listOf(1), state)
        val manifest = DreamLiteAbi.Manifest(
            components = mapOf(
                "text_encoder" to DreamLiteAbi.Component(listOf(t("tokens","tokens")), listOf(t("c","conditioning"))),
                "unet" to DreamLiteAbi.Component(
                    listOf(t("sample","model_input"),t("time","timestep"),t("cond","conditioning"),t("mask","attention_mask"),t("ids","time_ids")),
                    listOf(t("noise","model_output")),
                ),
                "vae_decoder" to DreamLiteAbi.Component(listOf(t("latent","latent")), listOf(t("image","image"))),
            ),
            scheduler = DreamLiteScheduler.Config(1000, true, "exponential"),
            vae = DreamLiteAbi.Vae(1f, 0f),
        )
        val conditioner = object : DreamLiteConditioner {
            override fun encode(request: DreamLiteConditioning.Request) =
                DreamLiteConditioner.Output(FloatArray(2048), floatArrayOf(1f), 1)
            override fun close() = Unit
        }
        val image = DreamLiteGeneration.run(
            runtime, manifest, DreamLiteGeneration.Request("portrait", 8, 8, 7), conditioner
        )
        assertEquals(listOf("unet","unet","unet","unet","vae_decoder"), calls)
        assertEquals(4, image.size)
    }
}
