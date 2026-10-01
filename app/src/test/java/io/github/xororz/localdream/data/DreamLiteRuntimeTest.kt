package io.github.xororz.localdream.data

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class DreamLiteRuntimeTest {
    @get:Rule
    val temp = TemporaryFolder()

    @Test
    fun reportsMissingQnnRuntimeLibraries() {
        val dir = temp.newFolder("runtime")
        assertEquals(
            listOf("libQnnHtp.so", "libQnnSystem.so"),
            DreamLiteRuntimeFactory.missingQnnLibraries(dir),
        )
        File(dir, "libQnnHtp.so").writeBytes(byteArrayOf(1))
        assertEquals(
            listOf("libQnnSystem.so"),
            DreamLiteRuntimeFactory.missingQnnLibraries(dir),
        )
    }

    @Test
    fun rejectsPackageThatChangesAfterProbe() {
        val dir = temp.newFolder("model")
        val files = listOf("unet.tflite", "ve.tflite", "vd.tflite", "te.tflite")
            .map { File(dir, it).apply { writeBytes(byteArrayOf(1)) } }
        val modelPackage = DreamLiteLiteRt.Package(files[0], files[1], files[2], files[3])
        files[0].delete()

        assertTrue(
            DreamLiteRuntimeFactory.create(modelPackage) is
                DreamLiteRuntimeFactory.Result.Unavailable,
        )
    }
}
