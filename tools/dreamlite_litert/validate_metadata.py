#!/usr/bin/env python3
"""Validate converted DreamLite tensor metadata before ABI emission."""
import argparse,json
from pathlib import Path

CORE_COMPONENTS={"unet","vae_encoder","vae_decoder"}
TEXT_BACKEND="litert_text"
QWEN_BACKEND="qwen3_vl_gguf"
UNET_KEYS={"model_input","timestep","conditioning","attention_mask","time_ids"}

def by_key(xs): return {x["state_key"]:x for x in xs}

def validate(meta, conditioning_backend=TEXT_BACKEND):
    if conditioning_backend not in {TEXT_BACKEND,QWEN_BACKEND}:
        raise ValueError("unsupported conditioning backend")
    comps=meta.get("components",{})
    required=set(CORE_COMPONENTS)
    if conditioning_backend==TEXT_BACKEND: required.add("text_encoder")
    missing=required-set(comps)
    if missing: raise ValueError("missing components: "+", ".join(sorted(missing)))

    u=comps["unet"];ins=by_key(u.get("inputs",[]));outs=by_key(u.get("outputs",[]))
    if not UNET_KEYS <= set(ins): raise ValueError("UNet semantic inputs do not match DreamLite Mobile")
    if "model_output" not in outs: raise ValueError("UNet model_output is missing")
    sample=ins["model_input"]["shape"];noise=outs["model_output"]["shape"]
    if len(sample)!=4 or len(noise)!=4: raise ValueError("UNet sample/noise must be rank-4 NCHW")
    if sample!=noise: raise ValueError("UNet input/output converted shapes must match")
    if sample[1]!=4: raise ValueError("DreamLite Mobile latent channel count must be 4")
    if sample[-1]%2: raise ValueError("DreamLite model_input width must contain two equal spatial halves")
    hidden=ins["conditioning"]["shape"]
    attention=ins["attention_mask"]["shape"]
    if hidden != [1,512,2048]:
        raise ValueError("DreamLite Mobile conditioning must be fixed [1,512,2048]")
    if attention != [1,512]:
        raise ValueError("DreamLite Mobile attention_mask must be fixed [1,512]")
    if ins["time_ids"]["shape"] != [1,2]: raise ValueError("DreamLite Mobile time_ids must be [1,2]")

    ve=comps["vae_encoder"];ve_out=by_key(ve.get("outputs",[]))
    vd=comps["vae_decoder"];vd_in=by_key(vd.get("inputs",[]))
    if "reference_latent" not in ve_out: raise ValueError("VAE encoder must produce reference_latent")
    if "latent" not in vd_in: raise ValueError("VAE decoder must consume latent")
    ref_shape=ve_out["reference_latent"]["shape"];dec_shape=vd_in["latent"]["shape"]
    if ref_shape != dec_shape: raise ValueError("VAE encoder/decoder latent shapes must match")
    if len(ref_shape)!=4 or ref_shape[1]!=4: raise ValueError("DreamLite VAE latent must be rank-4 with 4 channels")
    if sample[0]!=ref_shape[0] or sample[1]!=ref_shape[1] or sample[2]!=ref_shape[2] or sample[3]!=ref_shape[3]*2:
        raise ValueError("UNet model_input must be latent plus reference_latent concatenated by width")

    if conditioning_backend==TEXT_BACKEND:
        te=comps["text_encoder"];te_in=by_key(te.get("inputs",[]));te_out=by_key(te.get("outputs",[]))
        if "tokens" not in te_in or "conditioning" not in te_out or "attention_mask" not in te_out:
            raise ValueError("text encoder semantic routes are invalid")
    return meta

def main():
    p=argparse.ArgumentParser()
    p.add_argument("metadata")
    p.add_argument("--conditioning-backend",choices=[TEXT_BACKEND,QWEN_BACKEND],default=TEXT_BACKEND)
    a=p.parse_args()
    validate(json.loads(Path(a.metadata).read_text()),a.conditioning_backend)

if __name__=="__main__": main()
