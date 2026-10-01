package io.github.xororz.localdream.data

import android.content.Context
import android.os.Build
import java.io.File

/**
 * Validates the experimental DreamLite/LiteRT package independently from the
 * legacy QNN backend. Actual LiteRT model execution is added only after the
 * converted graph ABI is frozen against the PyTorch reference.
 */
object DreamLiteLiteRt {
    const val RUNTIME = "dreamlite_litert"

    /**
     * The experimental Qualcomm NPU path is intentionally narrower than the
     * generic LiteRT device matrix. SM8650 is Snapdragon 8 Gen 3 (Galaxy S24
     * Ultra target); newer SM parts are admitted for forward testing.
     */
    private const val FIRST_QUALCOMM_NPU_PART = 8650

    fun isSupportedNpuDevice(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return false
        val soc = Build.SOC_MODEL.uppercase()
        if (!soc.startsWith("SM")) return false
        val part = soc.dropWhile { !it.isDigit() }
            .takeWhile { it.isDigit() }
            .toIntOrNull()
        return part != null && part >= FIRST_QUALCOMM_NPU_PART
    }

    data class Package(
        val unet: File,
        val vaeEncoder: File,
        val vaeDecoder: File,
        val textEncoder: File,
        val abiManifest: File,
    ) {
        val files: List<File>
            get() = listOf(unet, vaeEncoder, vaeDecoder, textEncoder, abiManifest)
    }

    sealed interface ProbeResult {
        data class Ready(val modelPackage: Package) : ProbeResult
        data class Invalid(val reason: String) : ProbeResult
    }

    fun probe(context: Context, model: Model): ProbeResult {
        if (model.configDefaults.runtime != RUNTIME) {
            return ProbeResult.Invalid("model does not declare $RUNTIME")
        }
        val dir = File(Model.getModelsDir(context), model.id)
        return probe(dir, model.configDefaults)
    }

    fun probe(modelDir: File, config: ModelConfig): ProbeResult {
        if (config.runtime != RUNTIME) {
            return ProbeResult.Invalid("config does not declare $RUNTIME")
        }

        fun component(value: String?): File? {
            if (value.isNullOrBlank()) return null
            val file = File(modelDir, value)
            val root = runCatching { modelDir.canonicalFile }.getOrNull()
                ?: return null
            val canonical = runCatching { file.canonicalFile }.getOrNull()
                ?: return null
            // Package metadata must not escape the model directory.
            if (!canonical.toPath().startsWith(root.toPath())) return null
            return canonical.takeIf { it.isFile && it.length() > 0L }
        }

        val unet = component(config.dreamliteUnet)
            ?: return ProbeResult.Invalid("missing or invalid DreamLite U-Net")
        val vaeEncoder = component(config.dreamliteVaeEncoder)
            ?: return ProbeResult.Invalid("missing or invalid DreamLite VAE encoder")
        val vaeDecoder = component(config.dreamliteVaeDecoder)
            ?: return ProbeResult.Invalid("missing or invalid DreamLite VAE decoder")
        val textEncoder = component(config.dreamliteTextEncoder)
            ?: return ProbeResult.Invalid("missing or invalid DreamLite text encoder")
        val abiManifest = component(DreamLiteAbi.MANIFEST)
            ?: return ProbeResult.Invalid("missing or invalid ${DreamLiteAbi.MANIFEST}")
        when (val abi = DreamLiteAbi.parse(abiManifest)) {
            is DreamLiteAbi.ParseResult.Invalid -> return ProbeResult.Invalid(abi.reason)
            is DreamLiteAbi.ParseResult.Valid -> Unit
        }

        return ProbeResult.Ready(
            Package(unet, vaeEncoder, vaeDecoder, textEncoder, abiManifest),
        )
    }
}
