package io.github.xororz.localdream.data
import org.junit.Assert.*
import org.junit.Test

class DreamLiteConditionerTest {
 @Test fun outputIsWrittenToUnetConditioningState() {
  val c=object:DreamLiteConditioner{
   override fun encode(request:DreamLiteConditioning.Request)=DreamLiteConditioner.Output(FloatArray(2*2048){1f},floatArrayOf(1f,1f),2)
   override fun close(){}
  }
  val state=DreamLiteOrchestrator.PipelineState()
  c.applyTo(state,DreamLiteConditioning.Request(DreamLiteConditioning.Mode.GENERATE,"test"))
  assertEquals(4096,state.tensors[DreamLiteOrchestrator.CONDITIONING_STATE_KEY]!!.size)
  assertArrayEquals(floatArrayOf(1f,1f),state.tensors[DreamLiteOrchestrator.ATTENTION_MASK_STATE_KEY]!!,0f)
 }
 @Test(expected=IllegalArgumentException::class)
 fun rejectsWrongHiddenWidth(){ DreamLiteConditioner.Output(FloatArray(4),floatArrayOf(1f),1,4) }
}
