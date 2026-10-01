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
        private const val MAX_JOBS = 500
        private const val MAX_REFERENCES_PER_JOB = 8
        private const val MAX_DIMENSION = 4096
        private const val MAX_STEPS = 200

        fun parse(raw: String): AutonomousAssetPlan = parse(JSONObject(raw))

        fun parse(json: JSONObject): AutonomousAssetPlan {
            val jobsJson = json.getJSONArray("jobs")
            require(jobsJson.length() in 1..MAX_JOBS) {
                "Autonomous asset plan must contain 1..$MAX_JOBS jobs"
            }

            val seenIds = mutableSetOf<String>()
            val jobs = buildList {
                for (index in 0 until jobsJson.length()) {
                    val item = jobsJson.getJSONObject(index)
                    val references = item.optJSONArray("reference_images") ?: JSONArray()
                    require(references.length() <= MAX_REFERENCES_PER_JOB) {
                        "Job ${index + 1} has too many reference images"
                    }

                    val id = item.optString("id").ifBlank { "asset-${index + 1}" }
                    require(seenIds.add(id)) { "Duplicate autonomous asset job id: $id" }

                    val prompt = item.getString("prompt").trim()
                    require(prompt.isNotEmpty()) { "Job ${index + 1} has an empty prompt" }

                    val width = item.optInt("width", 1024)
                    val height = item.optInt("height", 1024)
                    require(width in 64..MAX_DIMENSION && height in 64..MAX_DIMENSION) {
                        "Job $id dimensions must be 64..$MAX_DIMENSION"
                    }

                    val steps = item.optInt("steps", 28)
                    require(steps in 1..MAX_STEPS) { "Job $id steps must be 1..$MAX_STEPS" }

                    val cfg = item.optDouble("cfg", 7.0).toFloat()
                    require(cfg.isFinite() && cfg >= 0f) { "Job $id cfg must be finite and non-negative" }

                    val scheduler = item.optString("scheduler", "dpm").trim()
                    require(scheduler.isNotEmpty()) { "Job $id scheduler is empty" }

                    add(
                        Job(
                            id = id,
                            prompt = prompt,
                            negativePrompt = item.optString("negative_prompt", ""),
                            seed = if (item.has("seed") && !item.isNull("seed")) item.getLong("seed") else null,
                            width = width,
                            height = height,
                            steps = steps,
                            cfg = cfg,
                            scheduler = scheduler,
                            referenceImages = buildList {
                                for (refIndex in 0 until references.length()) {
                                    add(references.getString(refIndex))
                                }
                            },
                        )
                    )
                }
            }

            val modelId = json.getString("model_id").trim()
            require(modelId.isNotEmpty()) { "model_id is required" }

            return AutonomousAssetPlan(
                runId = json.optString("run_id").ifBlank { "run-${System.currentTimeMillis()}" },
                modelId = modelId,
                backendType = json.optString("backend_type").takeIf { it.isNotBlank() },
                maxRetries = json.optInt("max_retries", 2).coerceIn(0, 10),
                jobs = jobs,
            )
        }
    }
}