package io.github.xororz.localdream.data

/**
 * Frozen application-side contract for converted DreamLite components.
 *
 * The converter must emit this metadata beside the .tflite files before the
 * Android runner is allowed to execute them. Shapes are intentionally not
 * guessed here: they are learned from the reference export and then pinned.
 */
object DreamLiteAbi {
    const val VERSION = 1
    const val MANIFEST = "dreamlite_abi.json"
    const val EXPECTED_STEPS = 4

    val requiredComponents = setOf(
        "unet",
        "vae_encoder",
        "vae_decoder",
        "text_encoder",
    )
}
