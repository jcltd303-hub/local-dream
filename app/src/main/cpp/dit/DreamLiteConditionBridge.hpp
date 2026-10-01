#ifndef DREAMLITE_CONDITION_BRIDGE_HPP
#define DREAMLITE_CONDITION_BRIDGE_HPP

#include <cstdint>
#include <string>
#include <vector>

struct sd_ctx_t;

struct DreamLiteConditionResult {
  std::vector<float> hidden_states;
  std::vector<float> attention_mask;
  int sequence_length = 0;
  int hidden_size = 0;
};

/**
 * Adapter over stable-diffusion.cpp's Qwen3-VL conditioner.
 *
 * Deliberately has no fallback implementation: the vendored sd.cpp C API must
 * expose standalone conditioning before this can report success. Returning
 * fabricated/zero conditioning would make package validation pass while
 * producing invalid DreamLite generations.
 */
class DreamLiteConditionBridge {
 public:
  static bool supported();
  static bool encode(sd_ctx_t *ctx, const std::string &prompt,
                     const uint8_t *reference_rgb, int width, int height,
                     DreamLiteConditionResult *out, std::string *error);
};

#endif
