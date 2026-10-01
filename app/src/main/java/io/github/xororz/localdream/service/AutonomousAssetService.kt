package io.github.xororz.localdream.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Environment
import android.os.IBinder
import android.util.Log
import androidx.core.app.ContextCompat
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.AutonomousAssetPlan
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject

/**
 * Resumable foreground orchestrator for unattended local asset batches.
 */
class AutonomousAssetService : Service() {
    private val scope = CoroutineScope(Dispatchers.IO + Job())
    private val notificationManager by lazy {
        getSystemService(NOTIFICATION_SERVICE) as NotificationManager
    }

    @Volatile
    private var stopRequested = false

    companion object {
        const val ACTION_START = "io.github.xororz.localdream.action.START_AUTONOMOUS_ASSETS"
        const val ACTION_RESUME = "io.github.xororz.localdream.action.RESUME_AUTONOMOUS_ASSETS"
        const val ACTION_STOP = "io.github.xororz.localdream.action.STOP_AUTONOMOUS_ASSETS"
        const val EXTRA_PLAN_JSON = "plan_json"

        private const val CHANNEL_ID = "autonomous_asset_generation"
        private const val NOTIFICATION_ID = 47

        fun start(context: Context, plan: AutonomousAssetPlan) {
            val intent = Intent(context, AutonomousAssetService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_PLAN_JSON, plan.toJson().toString())
            ContextCompat.startForegroundService(context, intent)
        }

        fun resume(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, AutonomousAssetService::class.java).setAction(ACTION_RESUME),
            )
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, AutonomousAssetService::class.java).setAction(ACTION_STOP),
            )
        }
    }

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, notification("Preparing autonomous run"))

        when (intent?.action) {
            ACTION_STOP -> {
                stopRequested = true
                BackgroundGenerationService.stop(applicationContext)
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_START -> {
                val raw = intent.getStringExtra(EXTRA_PLAN_JSON)
                if (raw.isNullOrBlank()) {
                    Log.e("AutonomousAssets", "Missing plan_json")
                    stopSelf()
                    return START_NOT_STICKY
                }
                stopRequested = false
                scope.launch { execute(AutonomousAssetPlan.parse(raw), reset = true) }
            }
            ACTION_RESUME, null -> {
                stopRequested = false
                scope.launch {
                    val state = latestStateFile()
                    if (state == null) {
                        Log.i("AutonomousAssets", "No autonomous run to resume")
                        stopSelf()
                    } else {
                        val json = JSONObject(state.readText())
                        execute(AutonomousAssetPlan.parse(json.getJSONObject("plan")), reset = false)
                    }
                }
            }
        }
        return START_STICKY
    }

    private suspend fun execute(plan: AutonomousAssetPlan, reset: Boolean) {
        val runDir = runStateDir(plan.runId)
        val stateFile = File(runDir, "state.json")
        val outputDir = outputDir(plan.runId)

        if (reset || !stateFile.exists()) {
            persistState(stateFile, plan, completed = emptySet(), failed = emptyMap(), active = null)
        }

        val saved = JSONObject(stateFile.readText())
        val completed = mutableSetOf<String>().apply {
            val array = saved.optJSONArray("completed") ?: JSONArray()
            for (i in 0 until array.length()) add(array.getString(i))
        }
        val failed = linkedMapOf<String, String>().apply {
            val objectJson = saved.optJSONObject("failed") ?: JSONObject()
            objectJson.keys().forEach { key -> put(key, objectJson.getString(key)) }
        }

        for ((index, job) in plan.jobs.withIndex()) {
            if (stopRequested) break
            if (job.id in completed) continue

            var lastError: String? = null
            var succeeded = false
            for (attempt in 0..plan.maxRetries) {
                if (stopRequested) break

                notifyProgress("Asset ${index + 1}/${plan.jobs.size}: ${job.id} (attempt ${attempt + 1})")
                persistState(stateFile, plan, completed, failed, job.id)
                prepareReferences(job)

                BackgroundGenerationService.resetState()
                val generationIntent = Intent(applicationContext, BackgroundGenerationService::class.java).apply {
                    putExtra("prompt", job.prompt)
                    putExtra("negative_prompt", job.negativePrompt)
                    putExtra("model_id", plan.modelId)
                    plan.backendType?.let { putExtra("backend_type", it) }
                    putExtra("steps", job.steps)
                    putExtra("cfg", job.cfg)
                    job.seed?.let { putExtra("seed", it) }
                    putExtra("width", job.width)
                    putExtra("height", job.height)
                    putExtra("effective_width", job.width)
                    putExtra("effective_height", job.height)
                    putExtra("scheduler", job.scheduler)
                    putExtra("aspect_ratio", "1:1")
                    putExtra("has_reference_images", job.referenceImages.isNotEmpty())
                }
                ContextCompat.startForegroundService(applicationContext, generationIntent)

                when (val result = BackgroundGenerationService.generationState.first {
                    it is BackgroundGenerationService.GenerationState.Complete ||
                        it is BackgroundGenerationService.GenerationState.Error
                }) {
                    is BackgroundGenerationService.GenerationState.Complete -> {
                        val imageFile = File(outputDir, safeName(job.id) + ".png")
                        savePng(result.bitmap, imageFile)
                        writeAssetMetadata(
                            File(outputDir, safeName(job.id) + ".json"),
                            plan,
                            job,
                            result.seed,
                            imageFile,
                        )
                        BackgroundGenerationService.markBitmapConsumed()
                        completed += job.id
                        failed.remove(job.id)
                        persistState(stateFile, plan, completed, failed, null)
                        succeeded = true
                    }
                    is BackgroundGenerationService.GenerationState.Error -> {
                        lastError = result.message
                        failed[job.id] = result.message
                        persistState(stateFile, plan, completed, failed, job.id)
                    }
                    else -> Unit
                }
                // GenerationService emits its terminal state before stopSelf() has
                // completed. Do not start the next request until that service is
                // fully gone or its previous stopSelf() can tear down the new job.
                withTimeoutOrNull(10_000L) {
                    BackgroundGenerationService.isServiceRunning.first { running -> !running }
                }
                if (succeeded) break
                if (attempt < plan.maxRetries) delay((attempt + 1L) * 2_000L)
            }

            if (!succeeded && lastError != null) {
                Log.e("AutonomousAssets", "Job ${job.id} exhausted retries: $lastError")
            }
        }

        val done = completed.size == plan.jobs.size
        persistState(stateFile, plan, completed, failed, null, done)
        notifyProgress(if (done) "Autonomous asset run complete" else "Autonomous asset run stopped")
        stopSelf()
    }

    private fun prepareReferences(job: AutonomousAssetPlan.Job) {
        val file = File(applicationContext.filesDir, "dit_references.json")
        if (job.referenceImages.isEmpty()) {
            if (file.exists()) file.delete()
            return
        }
        file.writeText(JSONArray(job.referenceImages).toString())
    }

    private fun savePng(bitmap: Bitmap, file: File) {
        file.parentFile?.mkdirs()
        FileOutputStream(file).use { stream ->
            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                "Failed to encode ${file.name}"
            }
        }
    }

    private fun writeAssetMetadata(
        file: File,
        plan: AutonomousAssetPlan,
        job: AutonomousAssetPlan.Job,
        actualSeed: Long?,
        imageFile: File,
    ) {
        file.writeText(
            JSONObject().apply {
                put("run_id", plan.runId)
                put("job_id", job.id)
                put("prompt", job.prompt)
                put("negative_prompt", job.negativePrompt)
                put("model_id", plan.modelId)
                plan.backendType?.let { put("backend_type", it) }
                put("seed", actualSeed ?: JSONObject.NULL)
                put("width", job.width)
                put("height", job.height)
                put("steps", job.steps)
                put("cfg", job.cfg.toDouble())
                put("scheduler", job.scheduler)
                put("image_path", imageFile.absolutePath)
                put("created_at_ms", System.currentTimeMillis())
            }.toString(2)
        )
    }

    private fun persistState(
        file: File,
        plan: AutonomousAssetPlan,
        completed: Set<String>,
        failed: Map<String, String>,
        active: String?,
        done: Boolean = false,
    ) {
        file.parentFile?.mkdirs()
        val target = File(file.parentFile, file.name + ".tmp")
        target.writeText(
            JSONObject().apply {
                put("plan", plan.toJson())
                put("completed", JSONArray(completed.toList()))
                put("failed", JSONObject(failed))
                put("active", active ?: JSONObject.NULL)
                put("done", done)
                put("updated_at_ms", System.currentTimeMillis())
            }.toString(2)
        )
        if (!target.renameTo(file)) {
            file.writeText(target.readText())
            target.delete()
        }
    }

    private fun runStateDir(runId: String): File =
        File(applicationContext.filesDir, "autonomous-assets/${safeName(runId)}").apply { mkdirs() }

    private fun outputDir(runId: String): File {
        val root = getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: filesDir
        return File(root, "LocalDreamAutonomous/${safeName(runId)}").apply { mkdirs() }
    }

    private fun latestStateFile(): File? {
        val root = File(applicationContext.filesDir, "autonomous-assets")
        return root.listFiles()
            ?.map { File(it, "state.json") }
            ?.filter { it.isFile }
            ?.maxByOrNull { it.lastModified() }
    }

    private fun safeName(value: String): String =
        value.replace(Regex("[^A-Za-z0-9._-]+"), "_").take(120).ifBlank { "asset" }

    private fun createChannel() {
        notificationManager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Autonomous asset generation",
                NotificationManager.IMPORTANCE_LOW,
            )
        )
    }

    private fun notification(text: String): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()

    private fun notifyProgress(text: String) {
        notificationManager.notify(NOTIFICATION_ID, notification(text))
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
