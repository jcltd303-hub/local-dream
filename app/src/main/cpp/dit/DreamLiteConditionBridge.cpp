#include "DreamLiteConditionBridge.hpp"

#include "stable-diffusion.h"

bool DreamLiteConditionBridge::supported() { return true; }

bool DreamLiteConditionBridge::encode(sd_ctx_t *ctx, const std::string &prompt,
                                      const uint8_t *reference_rgb, int width,
                                      int height,
                                      DreamLiteConditionResult *out,
                                      std::string *error) {
  if (!ctx || !out) {
    if (error) *error = "invalid DreamLite conditioner arguments";
    return false;
  }

  sd_condition_params_t params{};
  params.prompt = prompt.c_str();
  if (reference_rgb && width > 0 && height > 0) {
    params.reference_image.width = width;
    params.reference_image.height = height;
    params.reference_image.channel = 3;
    params.reference_image.data = const_cast<uint8_t *>(reference_rgb);
  }

  sd_condition_output_t encoded{};
  if (!sd_encode_condition(ctx, &params, &encoded)) {
    if (error) *error = "Qwen3-VL conditioning failed";
    return false;
  }

  const size_t hidden_count =
      static_cast<size_t>(encoded.sequence_length) * encoded.hidden_size;
  out->hidden_states.assign(encoded.hidden_states,
                            encoded.hidden_states + hidden_count);
  out->attention_mask.assign(encoded.attention_mask,
                             encoded.attention_mask + encoded.sequence_length);
  out->sequence_length = encoded.sequence_length;
  out->hidden_size = encoded.hidden_size;
  sd_free_condition(&encoded);
  return true;
}
