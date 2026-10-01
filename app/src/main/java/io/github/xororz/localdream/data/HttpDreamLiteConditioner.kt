package io.github.xororz.localdream.data

import android.util.Base64
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONObject

/**
 * Calls the local native Qwen3-VL conditioner served by stable_diffusion_core.
 * The native process owns GGUF/VLM memory; the LiteRT diffusion runtime only
 * receives the resulting DreamLite hidden states and attention mask.
 */
class HttpDreamLiteConditioner(
    private val endpoint: String = "http://127.0.0.1:8081/dreamlite/condition",
) : DreamLiteConditioner {
    override fun encode(request: DreamLiteConditioning.Request): DreamLiteConditioner.Output {
        val prepared = DreamLiteConditioning.prepare(request)
        val body = JSONObject()
            .put("prompt", prepared.prompt)
        if (prepared.requiresVision) {
            body.put("reference_rgb", Base64.encodeToString(request.referenceRgb, Base64.NO_WRAP))
            body.put("reference_width", request.referenceWidth)
            body.put("reference_height", request.referenceHeight)
        }
        val connection = URL(endpoint).openConnection() as HttpURLConnection
        return try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 30_000
            connection.readTimeout = 120_000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            connection.outputStream.use { it.write(body.toString().toByteArray()) }
            val stream = if (connection.responseCode in 200..299) {
                connection.inputStream
            } else {
                connection.errorStream
            }
            val response = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            require(connection.responseCode in 200..299) {
                "DreamLite conditioner failed HTTP ${connection.responseCode}: $response"
            }
            val json = JSONObject(response)
            val hiddenJson = json.getJSONArray("hidden_states")
            val maskJson = json.getJSONArray("attention_mask")
            val hidden = FloatArray(hiddenJson.length()) { hiddenJson.getDouble(it).toFloat() }
            val mask = FloatArray(maskJson.length()) { maskJson.getDouble(it).toFloat() }
            DreamLiteConditioner.Output(
                hidden,
                mask,
                json.getInt("sequence_length"),
                json.getInt("hidden_size"),
            )
        } finally {
            connection.disconnect()
        }
    }

    override fun close() = Unit
}
