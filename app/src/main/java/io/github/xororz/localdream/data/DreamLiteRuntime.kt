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

    sealed interface Result {
        data class Available(val runtime: DreamLiteRuntime) : Result
        data class Unavailable(val reason: String) : Result
    }

    fun create(modelPackage: DreamLiteLiteRt.Package): Result {
        // Touch the package here so callers cannot accidentally treat a runtime
        // as independent of the exact files validated by DreamLiteLiteRt.probe.
        val files = modelPackage.files
        if (files.size != 4 || files.any { !it.isFile || it.length() <= 0L }) {
            return Result.Unavailable("DreamLite LiteRT package is no longer valid")
        }
        return Result.Unavailable(
            "LiteRT runner not linked yet; refusing CPU/QNN fallback",
        )
    }
}
