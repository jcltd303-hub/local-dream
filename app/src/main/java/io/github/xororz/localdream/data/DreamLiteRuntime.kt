package io.github.xororz.localdream.data

import java.io.Closeable
import java.io.File
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

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

    /** Executes one converted component. Float32 is the frozen ABI v1 data path. */
    fun runFloatComponent(component: String, inputs: Map<String, FloatArray>): Map<String, FloatArray>

    override fun close() = Unit
}

/**
 * LiteRT-backed runtime for the converted DreamLite component graphs.
 *
 * Creation is NPU-only and fail-closed. Generation remains gated until
 * CompiledModel tensor metadata has been inspected against DreamLite ABI v1.
 */
private class LiteRtDreamLiteRuntime(
    private val models: List<Pair<File, CompiledModel>>,
    private val manifest: DreamLiteAbi.Manifest,
    private val compileTime: Long,
) : DreamLiteRuntime {
    private fun tensor(
        name: String,
        type: com.google.ai.edge.litert.TensorType,
    ) = DreamLiteRuntime.Tensor(
        name = name,
        shape = type.layout?.dimensions ?: emptyList(),
        dataType = when (type.elementType) {
            com.google.ai.edge.litert.TensorType.ElementType.FLOAT -> "float32"
            com.google.ai.edge.litert.TensorType.ElementType.INT -> "int32"
            com.google.ai.edge.litert.TensorType.ElementType.INT8 -> "int8"
            com.google.ai.edge.litert.TensorType.ElementType.BOOLEAN -> "bool"
            com.google.ai.edge.litert.TensorType.ElementType.INT64 -> "int64"
        },
    )

    override fun inspect(): DreamLiteRuntime.Diagnostics =
        DreamLiteRuntime.Diagnostics(
            runtime = "litert",
            accelerator = "npu",
            components = models.map { (file, model) ->
                val name = file.nameWithoutExtension
                val abi = manifest.components[name]
                    ?: manifest.components.entries.firstOrNull {
                        name.contains(it.key, ignoreCase = true)
                    }?.value
                    ?: error("DreamLite ABI has no component for $name")
                DreamLiteRuntime.ComponentInfo(
                    file = file,
                    inputs = abi.inputs.map {
                        tensor(it.name, model.getInputTensorType(it.name))
                    },
                    outputs = abi.outputs.map {
                        tensor(it.name, model.getOutputTensorType(it.name))
                    },
                )
            },
            // LiteRT currently adds CPU when NPU-only is requested so partially
            // compiled graphs can still execute. Until delegation metrics prove
            // every node is accelerated, fail closed instead of claiming NPU-only.
            cpuFallback = true,
            compileTimeMs = compileTime,
        )

    override fun runFloatComponent(
        component: String,
        inputs: Map<String, FloatArray>,
    ): Map<String, FloatArray> {
        val pair = models.firstOrNull { (file, _) ->
            file.nameWithoutExtension == component ||
                file.nameWithoutExtension.contains(component, ignoreCase = true)
        } ?: error("DreamLite component $component is not compiled")
        val model = pair.second
        val abi = manifest.components[component]
            ?: error("DreamLite ABI has no component $component")
        require(abi.inputs.all { it.dataType.equals("float32", ignoreCase = true) }) {
            "DreamLite ABI v1 execution currently supports float32 inputs only"
        }
        require(abi.outputs.all { it.dataType.equals("float32", ignoreCase = true) }) {
            "DreamLite ABI v1 execution currently supports float32 outputs only"
        }
        require(inputs.keys == abi.inputs.map { it.name }.toSet()) {
            "DreamLite $component input names do not match ABI"
        }

        val inputBuffers = abi.inputs.associate { tensor ->
            tensor.name to model.createInputBuffer(tensor.name).also {
                it.writeFloat(inputs.getValue(tensor.name))
            }
        }
        val outputBuffers = abi.outputs.associate { tensor ->
            tensor.name to model.createOutputBuffer(tensor.name)
        }
        return try {
            model.run(inputBuffers, outputBuffers)
            outputBuffers.mapValues { (_, buffer) -> buffer.readFloat() }
        } finally {
            inputBuffers.values.forEach { runCatching { it.close() } }
            outputBuffers.values.forEach { runCatching { it.close() } }
        }
    }

    override fun close() {
        models.forEach { (_, model) -> model.close() }
    }
}

object DreamLiteRuntimeFactory {
    const val REQUIRED_ACCELERATOR = "npu"
    const val CACHE_DIR = "dreamlite_litert_cache"

    // Required at runtime for the Qualcomm LiteRT path. QNN core libraries are
    // already packaged by the app; the LiteRT dispatch/compiler libraries are
    // supplied by a LiteRT Qualcomm runtime bundle during QNN-enabled builds.
    val requiredQnnLibraries = listOf(
        "libQnnHtp.so",
        "libQnnSystem.so",
        "libLiteRtDispatch_Qualcomm.so",
        "libLiteRtCompilerPlugin_Qualcomm.so",
    )

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
        val manifest = when (val parsed = DreamLiteAbi.parse(modelPackage.abiManifest)) {
            is DreamLiteAbi.ParseResult.Valid -> parsed.manifest
            is DreamLiteAbi.ParseResult.Invalid -> return Result.Unavailable(parsed.reason)
        }
        val start = System.nanoTime()
        val compiled = mutableListOf<Pair<File, CompiledModel>>()
        return try {
            files.filter { it.extension == "tflite" }.forEach { file ->
                compiled += file to CompiledModel.create(
                    file.absolutePath,
                    CompiledModel.Options(Accelerator.NPU),
                )
            }
            Result.Available(
                LiteRtDreamLiteRuntime(
                    compiled,
                    manifest,
                    (System.nanoTime() - start) / 1_000_000L,
                ),
            )
        } catch (e: Exception) {
            compiled.forEach { (_, model) -> runCatching { model.close() } }
            Result.Unavailable("LiteRT NPU compile failed: ${e.message ?: e.javaClass.simpleName}")
        }
    }
}
