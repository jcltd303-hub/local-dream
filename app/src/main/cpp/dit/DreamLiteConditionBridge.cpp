#include "DreamLiteConditionBridge.hpp"

bool DreamLiteConditionBridge::supported() { return false; }

bool DreamLiteConditionBridge::encode(sd_ctx_t *, const std::string &,
                                      const uint8_t *, int, int,
                                      DreamLiteConditionResult *,
                                      std::string *error) {
  if (error) {
    *error =
        "vendored stable-diffusion.cpp does not expose standalone Qwen3-VL "
        "conditioning through its public C API";
  }
  return false;
}
