package io.github.xororz.localdream.data

import kotlin.math.exp

/** Reference math used by DreamLite Mobile's FlowMatchEulerDiscreteScheduler. */
object DreamLiteScheduler {
    data class Config(
        val numTrainTimesteps: Int,
        val useDynamicShifting: Boolean,
        val timeShiftType: String,
        val baseImageSeqLen: Int = 256,
        val maxImageSeqLen: Int = 4096,
        val baseShift: Float = 0.5f,
        val maxShift: Float = 1.16f,
    )

    fun calculateMu(imageSeqLen: Int, config: Config): Float {
        val m = (config.maxShift - config.baseShift) /
            (config.maxImageSeqLen - config.baseImageSeqLen)
        val b = config.baseShift - m * config.baseImageSeqLen
        return imageSeqLen * m + b
    }

    fun shiftedSigma(rawSigma: Float, mu: Float, config: Config): Float {
        require(rawSigma in 0f..1f)
        if (rawSigma == 1f) return 1f
        require(rawSigma > 0f)
        if (!config.useDynamicShifting) return rawSigma
        val odds = 1.0 / rawSigma - 1.0
        return when (config.timeShiftType) {
            "exponential" -> {
                val e = exp(mu.toDouble())
                (e / (e + odds)).toFloat()
            }
            "linear" -> (mu / (mu + odds)).toFloat()
            else -> error("Unsupported FlowMatch time_shift_type: " + config.timeShiftType)
        }
    }

    fun schedule(imageSeqLen: Int, config: Config): List<Float> {
        val mu = if (config.useDynamicShifting) calculateMu(imageSeqLen, config) else 0f
        return DreamLiteOrchestrator.DEFAULT_SIGMAS.map { shiftedSigma(it, mu, config) } + 0f
    }

    fun timestep(sigma: Float, config: Config): Float =
        sigma * config.numTrainTimesteps

    fun eulerStep(sample: FloatArray, modelOutput: FloatArray, sigma: Float, sigmaNext: Float): FloatArray {
        require(sample.size == modelOutput.size)
        val dt = sigmaNext - sigma
        return FloatArray(sample.size) { i -> sample[i] + dt * modelOutput[i] }
    }
}
