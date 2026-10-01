package io.github.xororz.localdream.data

import org.junit.Assert.*
import org.junit.Test

class DreamLiteConditioningTest {
 @Test fun generationUsesOfficialTaskPrefixAndNoVision() {
  val p=DreamLiteConditioning.prepare(DreamLiteConditioning.Request(DreamLiteConditioning.Mode.GENERATE,"a red fox"))
  assertEquals("[Generate]: a red fox",p.prompt);assertFalse(p.requiresVision);assertEquals(34,p.dropPrefixTokens)
 }
 @Test fun editRequiresReferenceImage() {
  try { DreamLiteConditioning.prepare(DreamLiteConditioning.Request(DreamLiteConditioning.Mode.EDIT,"change coat"))
   fail("expected failure")
  } catch(e:IllegalArgumentException) { assertTrue(e.message!!.contains("reference image")) }
 }
 @Test fun editUsesOfficialDiptychInstruction() {
  val p=DreamLiteConditioning.prepare(DreamLiteConditioning.Request(DreamLiteConditioning.Mode.EDIT,"change coat",byteArrayOf(1),256,256))
  assertTrue(p.prompt.startsWith("[Edit]: A diptych with two side-by-side images of the same scene."))
  assertTrue(p.prompt.endsWith("change coat"));assertTrue(p.requiresVision);assertEquals(64,p.dropPrefixTokens)
 }
}
