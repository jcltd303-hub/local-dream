package io.github.xororz.localdream.data

object DreamLiteConditioning {
    enum class Mode { GENERATE, EDIT }

    data class Request(
        val mode: Mode,
        val prompt: String,
        val referenceRgb: ByteArray? = null,
        val referenceWidth: Int = 0,
        val referenceHeight: Int = 0,
    )

    data class Prepared(
        val prompt: String,
        val requiresVision: Boolean,
    )

    fun prepare(request: Request): Prepared {
        require(request.prompt.isNotBlank()) { "DreamLite prompt must not be blank" }
        return when (request.mode) {
            Mode.GENERATE -> Prepared("[Generate]: ${request.prompt}", false)
            Mode.EDIT -> {
                require(request.referenceRgb != null && request.referenceRgb.isNotEmpty()) {
                    "DreamLite edit conditioning requires a reference image"
                }
                require(request.referenceWidth > 0 && request.referenceHeight > 0) {
                    "DreamLite reference dimensions must be positive"
                }
                Prepared(
                    "[Edit]: A diptych with two side-by-side images of the same scene. " +
                        "Compared to the right side, the left one has ${request.prompt}",
                    true,
                )
            }
        }
    }
}
