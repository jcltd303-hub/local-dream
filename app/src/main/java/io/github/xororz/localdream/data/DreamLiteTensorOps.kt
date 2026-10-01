package io.github.xororz.localdream.data

/** Tensor transforms mirrored from the official DreamLite Mobile pipeline. */
object DreamLiteTensorOps {
    fun timeIds(width: Int, height: Int): FloatArray {
        require(width > 0 && height > 0)
        return floatArrayOf(width.toFloat(), height.toFloat())
    }

    /**
     * Concatenate NCHW tensors along width, matching torch.cat([latents, imageLatents], dim=3).
     */
    fun concatWidth(
        left: FloatArray,
        right: FloatArray,
        batch: Int,
        channels: Int,
        height: Int,
        width: Int,
    ): FloatArray {
        val plane = height * width
        val expected = batch * channels * plane
        require(left.size == expected && right.size == expected)
        val outWidth = width * 2
        val out = FloatArray(batch * channels * height * outWidth)
        for (b in 0 until batch) {
            for (c in 0 until channels) {
                for (y in 0 until height) {
                    val src = ((b * channels + c) * height + y) * width
                    val dst = ((b * channels + c) * height + y) * outWidth
                    left.copyInto(out, dst, src, src + width)
                    right.copyInto(out, dst + width, src, src + width)
                }
            }
        }
        return out
    }

    /** Official pipeline discards the conditioning half of UNet width before scheduler.step. */
    fun cropNoiseToLatentWidth(
        noise: FloatArray,
        batch: Int,
        channels: Int,
        height: Int,
        modelWidth: Int,
        latentWidth: Int,
    ): FloatArray {
        require(modelWidth >= latentWidth)
        require(noise.size == batch * channels * height * modelWidth)
        val out = FloatArray(batch * channels * height * latentWidth)
        for (b in 0 until batch) {
            for (c in 0 until channels) {
                for (y in 0 until height) {
                    val src = ((b * channels + c) * height + y) * modelWidth
                    val dst = ((b * channels + c) * height + y) * latentWidth
                    noise.copyInto(out, dst, src, src + latentWidth)
                }
            }
        }
        return out
    }

    fun imageSequenceLength(latentHeight: Int, latentWidth: Int): Int {
        require(latentHeight > 0 && latentWidth > 0)
        return latentHeight * latentWidth / 4
    }
}
