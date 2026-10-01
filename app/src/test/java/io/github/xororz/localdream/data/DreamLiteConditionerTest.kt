package io.github.xororz.localdream.data
import org.junit.Assert.*
import org.junit.Test

class DreamLiteConditionerTest {
 @Test fun outputIsPaddedToStaticUnetConditioningState() {
  val c=object:DreamLiteConditioner{
   override fun encode(request:DreamLiteConditioning.Request)=
    DreamLiteConditioner.Output(FloatArray(2*DreamLiteConditioner.HIDDEN_SIZE){1f},floatArrayOf(1f,1f),2)
  }
  val state=DreamLiteOrchestrator.PipelineState()
  c.applyTo(state,DreamLiteConditioning.Request(DreamLiteConditioning.Mode.GENERATE,"test"))
  val hidden=state.tensors.getValue(DreamLiteOrchestrator.CONDITIONING_STATE_KEY)
  val mask=state.tensors.getValue(DreamLiteOrchestrator.ATTENTION_MASK_STATE_KEY)
  assertEquals(DreamLiteConditioner.SEQUENCE_LENGTH*DreamLiteConditioner.HIDDEN_SIZE,hidden.size)
  assertEquals(DreamLiteConditioner.SEQUENCE_LENGTH,mask.size)
  assertEquals(1f,hidden[0],0f)
  assertEquals(1f,hidden[2*DreamLiteConditioner.HIDDEN_SIZE-1],0f)
  assertEquals(0f,hidden[2*DreamLiteConditioner.HIDDEN_SIZE],0f)
  assertArrayEquals(floatArrayOf(1f,1f),mask.copyOfRange(0,2),0f)
  assertEquals(0f,mask[2],0f)
 }
 @Test(expected=IllegalArgumentException::class)
 fun rejectsWrongHiddenWidth(){ DreamLiteConditioner.Output(FloatArray(4),floatArrayOf(1f),1,4) }
 @Test(expected=IllegalArgumentException::class)
 fun rejectsSequenceLongerThanStaticAbi(){
  DreamLiteConditioner.Output(
   FloatArray((DreamLiteConditioner.SEQUENCE_LENGTH+1)*DreamLiteConditioner.HIDDEN_SIZE),
   FloatArray(DreamLiteConditioner.SEQUENCE_LENGTH+1),
   DreamLiteConditioner.SEQUENCE_LENGTH+1
  )
 }
}
