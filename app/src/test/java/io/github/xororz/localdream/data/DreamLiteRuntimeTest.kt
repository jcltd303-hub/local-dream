package io.github.xororz.localdream.data

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class DreamLiteRuntimeTest {
    private fun tensor(name: String, shape: List<Int>) =
        DreamLiteRuntime.Tensor(name, shape, "float32")

    @get:Rule
    val temp = TemporaryFolder()

    @Test
    fun reportsMissingQnnRuntimeLibraries() {
        val dir = temp.newFolder("runtime")
        assertEquals(
            DreamLiteRuntimeFactory.requiredQnnLibraries,
            DreamLiteRuntimeFactory.missingQnnLibraries(dir),
        )
        File(dir, "libQnnHtp.so").writeBytes(byteArrayOf(1))
        assertEquals(
            DreamLiteRuntimeFactory.requiredQnnLibraries.drop(1),
            DreamLiteRuntimeFactory.missingQnnLibraries(dir),
        )
    }

    @Test
    fun acceptsMatchingNpuDiagnosticsWithDynamicManifestShape() {
        val manifest = DreamLiteAbi.Manifest(
            DreamLiteAbi.requiredComponents.associateWith {
                DreamLiteAbi.Component(
                    inputs = listOf(DreamLiteAbi.Tensor("in", "float32", listOf(1, -1))),
                    outputs = listOf(DreamLiteAbi.Tensor("out", "float32", listOf(1, 4))),
                )
            },
        )
        val diagnostics = DreamLiteRuntime.Diagnostics(
            runtime = "litert",
            accelerator = "npu",
            components = DreamLiteAbi.requiredComponents.map {
                DreamLiteRuntime.ComponentInfo(
                    file = File("$it.tflite"),
                    inputs = listOf(tensor("in", listOf(1, 77))),
                    outputs = listOf(tensor("out", listOf(1, 4))),
                )
            },
            cpuFallback = false,
        )
        assertEquals(null, DreamLiteRuntimeFactory.validateDiagnostics(manifest, diagnostics))
    }

    @Test
    fun recordsCpuFallbackCapabilityWithoutBlockingSoftwareReadiness() {
        val manifest = DreamLiteAbi.Manifest(emptyMap())
        val diagnostics = DreamLiteRuntime.Diagnostics(
            runtime = "litert",
            accelerator = "npu",
            components = emptyList(),
            cpuFallback = true,
        )
        assertEquals(null, DreamLiteRuntimeFactory.validateDiagnostics(manifest, diagnostics))
        assertTrue(diagnostics.cpuFallback)
    }

    @Test
    fun rejectsPackageThatChangesAfterProbe() {
        val dir = temp.newFolder("model")
        val files = listOf(
            "unet.tflite",
            "ve.tflite",
            "vd.tflite",
            "te.tflite",
            DreamLiteAbi.MANIFEST,
        ).map { File(dir, it).apply { writeBytes(byteArrayOf(1)) } }
        val modelPackage = DreamLiteLiteRt.Package(
            files[0],
            files[1],
            files[2],
            files[3],
            files[4],
        )
        files[0].delete()

        assertTrue(
            DreamLiteRuntimeFactory.create(modelPackage) is
                DreamLiteRuntimeFactory.Result.Unavailable,
        )
    }
}
