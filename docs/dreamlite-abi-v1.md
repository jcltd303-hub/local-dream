# DreamLite LiteRT ABI v1

This file defines the application-side acceptance contract for converted
DreamLite models. It deliberately does not invent tensor shapes from the
upstream PyTorch model.

A converted package must contain:

- `dreamlite_unet.tflite`
- `dreamlite_vae_encoder.tflite`
- `dreamlite_vae_decoder.tflite`
- `dreamlite_text_encoder.tflite`
- `dreamlite_abi.json`

The manifest must declare `abi_version: 1`, `runtime: "dreamlite_litert"`,
`steps: 4`, and input/output tensor metadata for every component. Each tensor
entry records name, dtype, and full shape.

## Acceptance rules

1. The Android app validates all package paths remain inside the model folder.
2. Components must exist and be non-empty.
3. Tensor names, dtypes, and shapes are compared with the manifest before the
   first generation.
4. The runtime records the selected accelerator.
5. Research benchmarking rejects a run when CPU fallback occurs.
6. The four-step denoising loop is enabled only after converted component
   outputs match deterministic PyTorch reference fixtures within documented
   tolerances.
7. Reference-image editing is wired only after the converted DreamLite edit
   conditioning tensors are confirmed from the reference implementation.

## Why shapes are not hard-coded yet

DreamLite conversion is still the experimental boundary. Guessing tensor
layouts would make a package appear supported while potentially feeding image,
conditioning, timestep, or latent data into the wrong axes. The conversion
tooling is responsible for generating the manifest from the exported graphs;
Android consumes and validates that manifest.
