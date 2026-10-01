#!/usr/bin/env python3
"""Emit the Android DreamLite ABI from converter-inspected tensor metadata.

This intentionally requires scheduler/VAE values from the source checkpoint.
It does not guess model constants.
"""
import argparse, json
from pathlib import Path

REQUIRED = ("unet", "vae_encoder", "vae_decoder", "text_encoder")

def tensor(name, dtype, shape, state_key):
    if not name or not dtype or not shape or any(int(x) == 0 or int(x) < -1 for x in shape):
        raise ValueError(f"invalid tensor {name}")
    return {"name": name, "dtype": dtype, "shape": [int(x) for x in shape], "state_key": state_key}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata", required=True, help="JSON produced by converted-model inspection")
    p.add_argument("--checkpoint-config", required=True, help="JSON containing scheduler and VAE source config")
    p.add_argument("--output", required=True)
    a=p.parse_args()
    meta=json.loads(Path(a.metadata).read_text())
    cfg=json.loads(Path(a.checkpoint_config).read_text())
    components={}
    for name in REQUIRED:
        src=meta["components"][name]
        components[name]={
            "inputs":[tensor(t["name"],t["dtype"],t["shape"],t["state_key"]) for t in src["inputs"]],
            "outputs":[tensor(t["name"],t["dtype"],t["shape"],t["state_key"]) for t in src["outputs"]],
        }
    scheduler=cfg["scheduler"]
    vae=cfg["vae"]
    required_scheduler=("num_train_timesteps","use_dynamic_shifting","time_shift_type",
                        "base_image_seq_len","max_image_seq_len","base_shift","max_shift")
    missing=[k for k in required_scheduler if k not in scheduler]
    if missing: raise ValueError("missing scheduler values: "+", ".join(missing))
    if "scaling_factor" not in vae:
        raise ValueError("missing VAE scaling_factor")
    vae.setdefault("shift_factor",0.0)
    out={"abi_version":1,"runtime":"dreamlite_litert","steps":4,
         "scheduler":{k:scheduler[k] for k in required_scheduler},
         "vae":{"scaling_factor":vae["scaling_factor"],"shift_factor":vae["shift_factor"]},
         "components":components}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
