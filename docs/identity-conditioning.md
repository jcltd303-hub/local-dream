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


## Locked first target: SD 1.5 base IP-Adapter

The first external adapter target is the canonical SD 1.5 IP-Adapter
(`ip-adapter_sd15.bin`), not InstantID and not IP-Adapter Plus.

Contract verified against Tencent's reference implementation:

- image encoder: OpenCLIP ViT-H/14 projection model
- encoder input preprocessing: CLIP image preprocessing
- encoder result: global projected image embedding
- image projection: Linear(1024 -> 4 * 768) + LayerNorm(768)
- image prompt tokens: 4 x 768 for SD 1.5
- text context remains the normal 77 x 768 CLIP context
- each UNet cross-attention layer owns separate learned image `to_k_ip` and
  `to_v_ip` projections
- adapter scale multiplies the image-attention contribution at the attention
  residual, not the CLIP embedding and not the final noise prediction

Therefore the stock Local Dream SD1.5 QNN UNet cannot be made IP-Adapter
compatible by changing its existing 77-token `encoder_hidden_states` input.
Its compiled graph currently has exactly three inputs: latent, timestep, and
77x768 text context.

### Required compiled UNet ABI

The re-exported SD1.5 QNN UNet must preserve the existing inputs and add one
named input carrying the four projected image tokens:

```
sample                  [1,4,H/8,W/8]
timestep                [1]
encoder_hidden_states   [1,77,768]
ip_adapter_tokens       [1,4,768]
ip_adapter_scale        [1]       # preferred; alternatively bake scale=1 and
                                  # scale image-attention outputs in graph
```

Every cross-attention block computes its normal text attention plus the
IP-Adapter K/V attention from `ip_adapter_tokens`. `ip_adapter_scale` is
applied only to that second contribution.

The QNN runtime must bind these inputs by tensor name. Positional binding is
not acceptable for the adapter graph.

### Mobile split

To keep the S24 Ultra memory peak controlled, package the identity path as
three independently loadable contexts:

1. `vision_encoder.bin`: CLIP ViT-H image -> global image embedding.
2. `identity_adapter.bin`: global embedding -> [1,4,768] projected tokens.
3. adapter-compatible `unet.bin`: consumes text context plus projected image
   tokens and scale.

The first two run once per selected reference image. Their outputs are cached,
then their QNN contexts may be released before denoising if device measurements
show a meaningful memory win. The UNet consumes only the small projected token
buffer on every denoising step.

A model package is not advertised as externally identity-capable until all
three graph contracts match. The runtime must fail closed on tensor count/name
or element-count mismatch.
