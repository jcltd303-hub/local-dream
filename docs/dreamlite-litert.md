# DreamLite Mobile + LiteRT Android backend

Status: experimental Android port scaffold. This does not claim that the
official DreamLite release ships a LiteRT model.

## Why this backend

DreamLite Mobile is a 0.39B, four-step unified generation/editing model. Its
editing path uses spatial latent concatenation rather than an IP-Adapter side
network, so a reference image is a native model input. That is a better fit for
Local Dream's reference-image UI than retrofitting SD1.5 cross-attention.

The official repository currently publishes an iOS on-device deployment
reference. Android/LiteRT conversion is work that this project must perform.

## License boundary

The DreamLite source code is Apache-2.0, but the released model weights are
CC BY-NC 4.0 and the authors describe them as non-commercial/research-only.
Do not ship those weights in a commercial Local Dream/influencer product
without obtaining separate rights. Keep weights user-supplied/gated during
this prototype.

## Target package

A custom model directory advertises the experimental runtime with:

```json
{
  "runtime": "dreamlite_litert",
  "default_steps": 4,
  "dreamlite_unet": "dreamlite_unet.tflite",
  "dreamlite_vae_encoder": "dreamlite_vae_encoder.tflite",
  "dreamlite_vae_decoder": "dreamlite_vae_decoder.tflite",
  "dreamlite_text_encoder": "dreamlite_text_encoder.tflite"
}
```

Exact file splitting may change after conversion profiling. The runtime must
validate tensor names/shapes from the converted models rather than assume a
PyTorch layout.

## Port sequence

1. Use the official DreamLite Mobile Diffusers checkpoint as the numerical
   reference and freeze deterministic generation/edit test vectors.
2. Export the distilled U-Net and tiny VAE with fixed 1024x1024 mobile shapes.
3. Convert each exported graph with LiteRT tooling and compare component
   outputs against PyTorch before quantization.
4. Port/convert the Qwen3-VL conditioning path separately. This is a major
   component and must not be represented as merely a text tokenizer.
5. Add the LiteRT Android runtime and select the Qualcomm NPU delegate only
   when the installed device/runtime reports support; keep a diagnostic
   fallback for graph bring-up.
6. Reuse Local Dream's existing reference-image picker. In edit mode encode the
   source image and construct DreamLite's native target/source spatial latent
   conditioning; in generation mode use the model's target/blank layout.
7. Run exactly four denoising steps for DreamLite Mobile and decode with the
   tiny VAE.
8. Benchmark cold load, conditioning, each denoise step, decode, peak RSS and
   thermals on the Galaxy S24 Ultra.

## Acceptance gate

Do not mark this backend production-ready until both deterministic text-to-
image and edit test vectors are within the agreed numerical/image tolerance of
the PyTorch reference, the Android run completes without CPU fallback for the
intended accelerator path, and licensing permits the intended distribution.


## Verified Android execution route (2026-09)

The current Google LiteRT stack supports diffusion/vision `.tflite` graphs through
the CompiledModel API. Qualcomm HTP/NPU execution uses the Qualcomm dispatch
plugin with QNN libraries. LiteRT documents both host AOT compilation and
on-device JIT/AOT-cache modes. For the S24 research build we target on-device
AOT caching: the first load compiles/partitions the original `.tflite` graph,
then later loads reuse the cached context.

Runtime acceptance remains strict:
- request NPU explicitly;
- package/load the Qualcomm dispatch/compiler plugin and required QNN runtime;
- cache compiled contexts under app-private storage;
- record compilation and inference timings;
- reject benchmark results when required DreamLite partitions fall back to CPU.

This means LiteRT is a runtime/dispatch layer, not a replacement for Qualcomm
QNN on Snapdragon. The existing app QNN runtime assets may be reusable, but
the LiteRT Qualcomm dispatch/compiler plugin is an additional required runtime
component.
