#!/usr/bin/env python3
"""Emit the Android DreamLite ABI from converter-inspected tensor metadata."""
import argparse,json
from pathlib import Path

CORE=("unet","vae_encoder","vae_decoder")
TEXT_BACKEND="litert_text"
QWEN_BACKEND="qwen3_vl_gguf"

def tensor(name,dtype,shape,state_key):
    if not name or not dtype or not shape or any(int(x)==0 or int(x)<-1 for x in shape):
        raise ValueError(f"invalid tensor {name}")
    return {"name":name,"dtype":dtype,"shape":[int(x) for x in shape],"state_key":state_key}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",required=True)
    p.add_argument("--checkpoint-config",required=True)
    p.add_argument("--conditioning-backend",choices=[TEXT_BACKEND,QWEN_BACKEND],default=TEXT_BACKEND)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    meta=json.loads(Path(a.metadata).read_text())
    cfg=json.loads(Path(a.checkpoint_config).read_text())

    required=list(CORE)
    if a.conditioning_backend==TEXT_BACKEND: required.append("text_encoder")
    components={}
    for name in required:
        src=meta["components"][name]
        components[name]={
            "inputs":[tensor(t["name"],t["dtype"],t["shape"],t["state_key"]) for t in src["inputs"]],
            "outputs":[tensor(t["name"],t["dtype"],t["shape"],t["state_key"]) for t in src["outputs"]],
        }

    scheduler=cfg["scheduler"];vae=cfg["vae"]
    required_scheduler=("num_train_timesteps","use_dynamic_shifting","time_shift_type",
                        "base_image_seq_len","max_image_seq_len","base_shift","max_shift")
    missing=[k for k in required_scheduler if k not in scheduler]
    if missing: raise ValueError("missing scheduler values: "+", ".join(missing))
    if "scaling_factor" not in vae: raise ValueError("missing VAE scaling_factor")
    vae=dict(vae);vae.setdefault("shift_factor",0.0)
    out={
        "abi_version":1,
        "runtime":"dreamlite_litert",
        "steps":4,
        "conditioning_backend":a.conditioning_backend,
        "scheduler":{k:scheduler[k] for k in required_scheduler},
        "vae":{"scaling_factor":vae["scaling_factor"],"shift_factor":vae["shift_factor"]},
        "components":components,
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
