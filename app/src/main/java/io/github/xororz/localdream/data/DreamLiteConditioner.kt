package io.github.xororz.localdream.data

/**
 * Backend boundary for DreamLite's Qwen3-VL conditioning stage.
 *
 * The mobile NPU graph uses a fixed 512-token conditioning ABI. Native Qwen
 * returns the exact valid sequence; applyTo() pads hidden states and mask with
 * zeros so masked attention is equivalent without requiring a dynamic NPU graph.
 */
interface DreamLiteConditioner : AutoCloseable {
    companion object {
        const val HIDDEN_SIZE = 2048
        const val SEQUENCE_LENGTH = 512
    }

    data class Output(
        val hiddenStates: FloatArray,
        val attentionMask: FloatArray,
        val sequenceLength: Int,
        val hiddenSize: Int = HIDDEN_SIZE,
    ) {
        init {
            require(sequenceLength > 0)
            require(sequenceLength <= SEQUENCE_LENGTH) {
                "DreamLite conditioning exceeds $SEQUENCE_LENGTH tokens"
            }
            require(hiddenSize == HIDDEN_SIZE)
            require(hiddenStates.size == sequenceLength * hiddenSize)
            require(attentionMask.size == sequenceLength)
        }
    }

    fun encode(request: DreamLiteConditioning.Request): Output

    fun applyTo(state: DreamLiteOrchestrator.PipelineState, request: DreamLiteConditioning.Request) {
        val output = encode(request)
        val hidden = FloatArray(SEQUENCE_LENGTH * HIDDEN_SIZE)
        output.hiddenStates.copyInto(hidden)
        val mask = FloatArray(SEQUENCE_LENGTH)
        output.attentionMask.copyInto(mask)
        state.tensors[DreamLiteOrchestrator.CONDITIONING_STATE_KEY] = hidden
        state.tensors[DreamLiteOrchestrator.ATTENTION_MASK_STATE_KEY] = mask
    }

    override fun close() = Unit
}
