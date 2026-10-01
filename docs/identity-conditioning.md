# Identity / Reference Conditioning

Local Dream now has two deliberately separate reference-image paths.

## Native DiT reference editing

FLUX.2/Klein and Qwen Image 2.1 use the model's native reference-image pathway.
The existing `reference_images` request field is decoded by `RequestParser.hpp`
and forwarded as clean reference images through `PipelineDit` and
`DitEngine`. Do not route these models through an IP-Adapter shim.

## External identity adapters (SD1.5 / SDXL)

A model package may declare:

```json
{
  "identity_vision_encoder": "vision_encoder.bin",
  "identity_adapter": "identity_adapter.bin",
  "identity_adapter_scale": 0.8
}
```

These fields advertise packaged adapter assets; they do not by themselves make
a stock QNN UNet adapter-compatible. The adapter graph must match the exported
UNet's attention/residual injection contract.

Implementation sequence:

1. Run `vision_encoder.bin` once before denoising and cache its output.
2. Validate the encoder output tensor shape against adapter metadata.
3. Run the adapter/projection graph to produce the exact tensors expected by
   the re-exported UNet.
4. Inject only through named adapter inputs in that UNet. Do not concatenate
   arbitrary image tokens onto the stock `encoder_hidden_states`; stock
   SD1.5/SDXL QNN graphs were compiled for fixed CLIP sequence shapes.
5. Release the vision context after embedding/projection when memory pressure
   requires it; retain only the conditioning tensors for the denoising loop.

The scale is clamped to 0..2 by Android config parsing. Runtime code must still
validate all graph tensor dimensions before execution.

## Device validation

Benchmark the vision-only pass separately before enabling generation. Record
initialization time, first/subsequent inference latency, peak memory, output
shape and embedding norm. Identity quality should be evaluated independently
from prompt adherence (for example with a face-embedding cosine metric over a
multi-pose/multi-lighting test set).
