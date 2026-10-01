#ifndef DREAMLITE_CONDITIONER_HOST_HPP
#define DREAMLITE_CONDITIONER_HOST_HPP

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "DitEngine.h"

class DreamLiteConditionerHost {
 public:
  ~DreamLiteConditionerHost() {
    if (ctx_ && api_) api_->destroy_conditioner(ctx_);
    if (handle_) dlclose(handle_);
  }

  bool initialize(const std::string &engine_path, const std::string &llm_path,
                  const std::string &vision_path, int n_threads = 4) {
    handle_ = dlopen(engine_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      error_ = std::string("dlopen conditioner engine failed: ") + dlerror();
      return false;
    }
    auto get_api = reinterpret_cast<dit_engine_get_api_fn>(
        dlsym(handle_, DIT_ENGINE_ENTRY_SYMBOL));
    if (!get_api) {
      error_ = std::string("conditioner engine entry point missing: ") + dlerror();
      return false;
    }
    api_ = get_api(DIT_ENGINE_ABI_VERSION);
    if (!api_) {
      error_ = "conditioner engine ABI mismatch";
      return false;
    }
    ctx_ = api_->create_conditioner(
        llm_path.c_str(), vision_path.empty() ? nullptr : vision_path.c_str(),
        "te=HTP0", "all=disk", n_threads);
    if (!ctx_) {
      error_ = "failed to create Qwen3-VL conditioner context";
      return false;
    }
    return true;
  }

  bool encode(const std::string &prompt, const uint8_t *rgb, int width,
              int height, int drop_prefix_tokens,
              std::vector<float> &hidden_states,
              std::vector<float> &attention_mask, int &sequence_length,
              int &hidden_size) {
    if (!ctx_ || !api_) {
      error_ = "conditioner host is not initialized";
      return false;
    }
    dit_condition_params params{};
    params.prompt = prompt.c_str();
    params.reference_image_rgb = rgb;
    params.reference_width = width;
    params.reference_height = height;
    params.drop_prefix_tokens = drop_prefix_tokens;

    dit_condition_output out{};
    if (!api_->condition_standalone(ctx_, &params, &out)) {
      const char *detail =
          api_->condition_last_error ? api_->condition_last_error(ctx_) : nullptr;
      error_ = detail && *detail ? detail : "Qwen3-VL conditioning failed";
      return false;
    }

    const size_t hidden_count =
        static_cast<size_t>(out.sequence_length) * out.hidden_size;
    hidden_states.assign(out.hidden_states, out.hidden_states + hidden_count);
    attention_mask.assign(out.attention_mask,
                          out.attention_mask + out.sequence_length);
    sequence_length = out.sequence_length;
    hidden_size = out.hidden_size;
    api_->free_condition(&out);
    return true;
  }

  const std::string &last_error() const { return error_; }

 private:
  void *handle_ = nullptr;
  const dit_engine_api *api_ = nullptr;
  dit_condition_ctx *ctx_ = nullptr;
  std::string error_;
};

#endif
