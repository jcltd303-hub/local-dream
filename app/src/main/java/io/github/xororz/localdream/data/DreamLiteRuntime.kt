package io.github.xororz.localdream.data

import java.io.Closeable
import java.io.File

/**
 * Execution boundary for the experimental DreamLite Android backend.
 *
 * Keeping this API free of LiteRT classes lets package/config tests run without
 * loading native accelerator libraries. The concrete LiteRT implementation can
 * be swapped as the converted graph ABI and Qualcomm delegate support mature.
 */
interface DreamLiteRuntime : Closeable {
    data class Tensor(
        val name: String,
        val shape: List<Int>,
        val dataType: String,
    )

    data class ComponentInfo(
        val file: File,
        val inputs: List<Tensor>,
        val outputs: List<Tensor>,
    )

    data class Diagnostics(
        val runtime: String,
        val accelerator: String,
        val components: List<ComponentInfo>,
        val cpuFallback: Boolean,
        val compileTimeMs: Long? = null,
        val cacheHit: Boolean? = null,
    )

    fun inspect(): Diagnostics

    override fun close() = Unit
}

/**
 * Placeholder factory deliberately fails closed until the LiteRT CompiledModel
 * dependency and converted DreamLite ABI are both pinned. No request may fall
 * back into the QNN Stable Diffusion executable.
 */
object DreamLiteRuntimeFactory {
    const val REQUIRED_ACCELERATOR = "npu"
    const val CACHE_DIR = "dreamlite_litert_cache"

    // Required at runtime for the Qualcomm LiteRT path. QNN core libraries are
    // already packaged by the app; the LiteRT dispatch/compiler libraries are
    // supplied by a LiteRT Qualcomm runtime bundle during QNN-enabled builds.
    val requiredQnnLibraries = listOf("libQnnHtp.so", "libQnnSystem.so")

    fun missingQnnLibraries(runtimeDir: File): List<String> =
        requiredQnnLibraries.filterNot { File(runtimeDir, it).isFile }

    fun validateDiagnostics(
        manifest: DreamLiteAbi.Manifest,
        diagnostics: DreamLiteRuntime.Diagnostics,
    ): String? {
        if (diagnostics.cpuFallback) return "DreamLite runtime used CPU fallback"
        if (!diagnostics.accelerator.equals(REQUIRED_ACCELERATOR, ignoreCase = true)) {
            return "DreamLite runtime selected ${diagnostics.accelerator}, expected $REQUIRED_ACCELERATOR"
        }

        val actualByName = diagnostics.components.associateBy { it.file.nameWithoutExtension }
        for ((name, expected) in manifest.components) {
            val actual = actualByName[name]
                ?: diagnostics.components.firstOrNull {
                    it.file.nameWithoutExtension.contains(name, ignoreCase = true)
                }
                ?: return "DreamLite runtime did not inspect component $name"

            fun compare(
                kind: String,
                expectedTensors: List<DreamLiteAbi.Tensor>,
                actualTensors: List<DreamLiteRuntime.Tensor>,
            ): String? {
                if (expectedTensors.size != actualTensors.size) {
                    return "$name $kind tensor count mismatch"
                }
                expectedTensors.zip(actualTensors).forEachIndexed { index, (e, a) ->
                    if (e.name != a.name || !e.dataType.equals(a.dataType, ignoreCase = true)) {
                        return "$name $kind tensor $index name/dtype mismatch"
                    }
                    if (e.shape.size != a.shape.size ||
                        e.shape.zip(a.shape).any { (ed, ad) -> ed != -1 && ed != ad }
                    ) {
                        return "$name $kind tensor $index shape mismatch"
                    }
                }
                return null
            }

            compare("input", expected.inputs, actual.inputs)?.let { return it }
            compare("output", expected.outputs, actual.outputs)?.let { return it }
        }
        return null
    }

    sealed interface Result {
        data class Available(val runtime: DreamLiteRuntime) : Result
        data class Unavailable(val reason: String) : Result
    }

    fun create(modelPackage: DreamLiteLiteRt.Package): Result {
        // Touch the package here so callers cannot accidentally treat a runtime
        // as independent of the exact files validated by DreamLiteLiteRt.probe.
        val files = modelPackage.files
        if (files.size != 5 || files.any { !it.isFile || it.length() <= 0L }) {
            return Result.Unavailable("DreamLite LiteRT package is no longer valid")
        }
        return Result.Unavailable(
            "LiteRT runner not linked yet; refusing CPU/QNN fallback",
        )
    }
}
