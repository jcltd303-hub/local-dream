package io.github.xororz.localdream.data

import org.json.JSONArray
import org.json.JSONObject

/**
 * Persistent input contract for unattended local asset generation.
 *
 * A plan is intentionally model/runtime agnostic: it forwards the normal
 * BackgroundGenerationService parameters so the autonomous layer never owns
 * inference semantics.
 */
data class AutonomousAssetPlan(
    val runId: String,
    val modelId: String,
    val backendType: String?,
    val maxRetries: Int,
    val jobs: List<Job>,
) {
    data class Job(
        val id: String,
        val prompt: String,
        val negativePrompt: String = "",
        val seed: Long? = null,
        val width: Int = 1024,
        val height: Int = 1024,
        val steps: Int = 28,
        val cfg: Float = 7f,
        val scheduler: String = "dpm",
        val referenceImages: List<String> = emptyList(),
    )

    fun toJson(): JSONObject = JSONObject().apply {
        put("run_id", runId)
        put("model_id", modelId)
        backendType?.let { put("backend_type", it) }
        put("max_retries", maxRetries)
        put("jobs", JSONArray().apply {
            jobs.forEach { job ->
                put(JSONObject().apply {
                    put("id", job.id)
                    put("prompt", job.prompt)
                    put("negative_prompt", job.negativePrompt)
                    job.seed?.let { put("seed", it) }
                    put("width", job.width)
                    put("height", job.height)
                    put("steps", job.steps)
                    put("cfg", job.cfg.toDouble())
                    put("scheduler", job.scheduler)
                    put("reference_images", JSONArray(job.referenceImages))
                })
            }
        })
    }

    companion object {
        fun parse(raw: String): AutonomousAssetPlan = parse(JSONObject(raw))

        fun parse(json: JSONObject): AutonomousAssetPlan {
            val jobsJson = json.getJSONArray("jobs")
            require(jobsJson.length() > 0) { "Autonomous asset plan must contain at least one job" }

            val jobs = buildList {
                for (index in 0 until jobsJson.length()) {
                    val item = jobsJson.getJSONObject(index)
                    val references = item.optJSONArray("reference_images") ?: JSONArray()
                    add(
                        Job(
                            id = item.optString("id").ifBlank { "asset-${index + 1}" },
                            prompt = item.getString("prompt").trim().also {
                                require(it.isNotEmpty()) { "Job ${index + 1} has an empty prompt" }
                            },
                            negativePrompt = item.optString("negative_prompt", ""),
                            seed = if (item.has("seed") && !item.isNull("seed")) item.getLong("seed") else null,
                            width = item.optInt("width", 1024),
                            height = item.optInt("height", 1024),
                            steps = item.optInt("steps", 28),
                            cfg = item.optDouble("cfg", 7.0).toFloat(),
                            scheduler = item.optString("scheduler", "dpm"),
                            referenceImages = buildList {
                                for (refIndex in 0 until references.length()) {
                                    add(references.getString(refIndex))
                                }
                            },
                        )
                    )
                }
            }

            return AutonomousAssetPlan(
                runId = json.optString("run_id").ifBlank { "run-${System.currentTimeMillis()}" },
                modelId = json.getString("model_id"),
                backendType = json.optString("backend_type").takeIf { it.isNotBlank() },
                maxRetries = json.optInt("max_retries", 2).coerceIn(0, 10),
                jobs = jobs,
            )
        }
    }
}
