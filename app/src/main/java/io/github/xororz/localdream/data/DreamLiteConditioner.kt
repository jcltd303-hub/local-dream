package io.github.xororz.localdream.data

/**
 * Backend boundary for DreamLite's Qwen3-VL conditioning stage.
 *
 * Implementations may use LiteRT or the existing native GGUF/vision stack, but
 * must return the exact hidden-state/mask tensors consumed by DreamLite UNet.
 */
interface DreamLiteConditioner : AutoCloseable {
    data class Output(
        val hiddenStates: FloatArray,
        val attentionMask: FloatArray,
        val sequenceLength: Int,
        val hiddenSize: Int = 2048,
    ) {
        init {
            require(sequenceLength > 0)
            require(hiddenSize == 2048)
            require(hiddenStates.size == sequenceLength * hiddenSize)
            require(attentionMask.size == sequenceLength)
        }
    }

    fun encode(request: DreamLiteConditioning.Request): Output

    fun applyTo(state: DreamLiteOrchestrator.PipelineState, request: DreamLiteConditioning.Request) {
        val output = encode(request)
        state.tensors[DreamLiteOrchestrator.CONDITIONING_STATE_KEY] = output.hiddenStates
        state.tensors[DreamLiteOrchestrator.ATTENTION_MASK_STATE_KEY] = output.attentionMask
    }
}
