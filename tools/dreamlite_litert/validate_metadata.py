#!/usr/bin/env python3
"""Validate converted DreamLite tensor metadata before ABI emission."""
import argparse, json
from pathlib import Path

REQUIRED_COMPONENTS={"unet","vae_encoder","vae_decoder","text_encoder"}
UNET_KEYS={"model_input","timestep","conditioning","attention_mask","time_ids"}
def by_key(xs): return {x["state_key"]:x for x in xs}

def validate(meta):
    comps=meta.get("components",{})
    missing=REQUIRED_COMPONENTS-set(comps)
    if missing: raise ValueError("missing components: "+", ".join(sorted(missing)))
    u=comps["unet"]; ins=by_key(u.get("inputs",[])); outs=by_key(u.get("outputs",[]))
    if not UNET_KEYS <= set(ins): raise ValueError("UNet semantic inputs do not match DreamLite Mobile")
    if "model_output" not in outs: raise ValueError("UNet model_output is missing")
    sample=ins["model_input"]["shape"]; noise=outs["model_output"]["shape"]
    if len(sample)!=4 or len(noise)!=4: raise ValueError("UNet sample/noise must be rank-4 NCHW")
    if sample != noise: raise ValueError("UNet input/output converted shapes must match")
    if sample[1] != 4: raise ValueError("DreamLite Mobile latent channel count must be 4")
    if sample[-1] % 2: raise ValueError("DreamLite model_input width must contain two equal spatial halves")
    hidden=ins["conditioning"]["shape"]
    if len(hidden)!=3 or hidden[-1]!=2048: raise ValueError("DreamLite Mobile conditioning width must be 2048")
    if ins["time_ids"]["shape"] != [1,2]: raise ValueError("DreamLite Mobile time_ids must be [1,2]")
    return meta

def main():
    p=argparse.ArgumentParser(); p.add_argument("metadata"); a=p.parse_args()
    validate(json.loads(Path(a.metadata).read_text()))

if __name__=="__main__": main()
