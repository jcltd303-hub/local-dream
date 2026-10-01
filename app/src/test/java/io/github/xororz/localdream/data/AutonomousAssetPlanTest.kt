package io.github.xororz.localdream.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class AutonomousAssetPlanTest {
    @Test
    fun parsesBatchPlanWithDefaultsAndReferences() {
        val plan = AutonomousAssetPlan.parse(
            """
            {
              "run_id": "campaign-001",
              "model_id": "dreamlite-mobile",
              "backend_type": "dreamlite_litert",
              "max_retries": 3,
              "jobs": [
                {
                  "id": "hero",
                  "prompt": "editorial portrait",
                  "seed": 42,
                  "reference_images": ["ZmFrZQ=="]
                },
                {
                  "prompt": "streetwear variation"
                }
              ]
            }
            """.trimIndent()
        )

        assertEquals("campaign-001", plan.runId)
        assertEquals("dreamlite-mobile", plan.modelId)
        assertEquals("dreamlite_litert", plan.backendType)
        assertEquals(3, plan.maxRetries)
        assertEquals(2, plan.jobs.size)
        assertEquals("hero", plan.jobs[0].id)
        assertEquals(42L, plan.jobs[0].seed)
        assertEquals(listOf("ZmFrZQ=="), plan.jobs[0].referenceImages)
        assertEquals("asset-2", plan.jobs[1].id)
        assertNull(plan.jobs[1].seed)
        assertEquals(1024, plan.jobs[1].width)
    }
    @Test
    fun rejectsDuplicateJobIds() {
        assertThrows(IllegalArgumentException::class.java) {
            AutonomousAssetPlan.parse(
                """
                {
                  "model_id": "dreamlite-mobile",
                  "jobs": [
                    {"id":"same","prompt":"one"},
                    {"id":"same","prompt":"two"}
                  ]
                }
                """.trimIndent()
            )
        }
    }

    @Test
    fun rejectsInvalidDimensions() {
        assertThrows(IllegalArgumentException::class.java) {
            AutonomousAssetPlan.parse(
                """
                {
                  "model_id": "dreamlite-mobile",
                  "jobs": [
                    {"id":"bad","prompt":"one","width":32,"height":1024}
                  ]
                }
                """.trimIndent()
            )
        }
    }
}
