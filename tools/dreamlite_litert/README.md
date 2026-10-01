# DreamLite Mobile -> LiteRT package

This directory is the reproducible conversion boundary for the experimental Android backend.

## Build sequence

1. Create a Python 3.11 conversion environment and install `requirements.txt` plus DreamLite/diffusers dependencies.
2. Capture the actual checkpoint constants:
   `capture_checkpoint_config.py --model <DreamLite-mobile> --output checkpoint.json`
3. Convert the diffusion UNet:
   `convert_unet.py --model <DreamLite-mobile> --output components/dreamlite_unet.tflite --metadata components/unet.json`
4. Convert the Tiny VAE:
   `convert_vae.py --model <DreamLite-mobile> --output-dir components`
5. Provide the conditioning component metadata/model.
6. Assemble:
   `build_package.py --components components --checkpoint-config checkpoint.json --output package`

Every converted numerical component must pass PyTorch/LiteRT parity before it is emitted.

## Conditioning blocker

DreamLite Mobile does not use a plain text-only encoder for identity editing. The official pipeline uses Qwen3-VL:
- generation: tokenized text -> Qwen3-VL hidden states;
- editing: templated text + a 256x256 reference image -> Qwen3-VL hidden states;
- the resulting hidden states and attention mask feed the UNet;
- the reference image is independently VAE-encoded and concatenated spatially with the denoising latent.

Therefore `dreamlite_text_encoder.tflite` is a package-interface name, not permission to substitute a text-only encoder. A production edit-capable package must implement the Qwen3-VL multimodal conditioning contract and prove parity.

## Ready criteria

Software-ready means conversion tests and Android unit tests are green, all package components exist and pass ABI validation, and the APK can consume the package. Device-ready additionally requires a Galaxy S24 Ultra run proving Qualcomm NPU delegation without CPU fallback, with latency and memory measurements.
