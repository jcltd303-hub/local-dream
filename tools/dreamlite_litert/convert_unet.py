#!/usr/bin/env python3
"""Convert DreamLite Mobile UNet directly from PyTorch to LiteRT with parity gate."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import litert_torch

class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__(); self.unet=unet
    def forward(self, sample, timestep, encoder_hidden_states, encoder_attention_mask, time_ids):
        return self.unet(
            sample=sample, timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"time_ids":time_ids},
            return_dict=False,
        )[0]

def sample_inputs(seq=512, seed=0):
    g=torch.Generator().manual_seed(seed)
    return (
      torch.randn(1,4,128,256,generator=g),
      torch.tensor([500.0],dtype=torch.float32),
      torch.randn(1,seq,2048,generator=g),
      torch.ones(1,seq,dtype=torch.float32),
      torch.tensor([[1024.0,1024.0]],dtype=torch.float32),
    )

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--metadata",required=True)
    p.add_argument("--atol",type=float,default=5e-3)
    p.add_argument("--rtol",type=float,default=5e-3)
    a=p.parse_args()
    from dreamlite import DreamLiteMobilePipeline
    unet=DreamLiteMobilePipeline.from_pretrained(a.model,sub_folder="unet",dtype=torch.float32)
    unet.eval()
    wrapper=UNetWrapper(unet).eval()
    args=sample_inputs()
    with torch.no_grad(): reference=wrapper(*args).detach().cpu().numpy()
    edge=litert_torch.convert(wrapper,args)
    converted=edge(*args)
    if isinstance(converted,(list,tuple)): converted=converted[0]
    converted=np.asarray(converted)
    if converted.shape != reference.shape:
        raise RuntimeError(f"LiteRT shape mismatch: {converted.shape} != {reference.shape}")
    max_abs=float(np.max(np.abs(reference-converted)))
    if not np.allclose(reference,converted,atol=a.atol,rtol=a.rtol):
        raise RuntimeError(f"LiteRT parity failed: max_abs={max_abs}")
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    edge.export(a.output)
    meta={"component":"unet","parity":{"atol":a.atol,"rtol":a.rtol,"max_abs":max_abs},
      "inputs":[
        {"name":"sample","dtype":"float32","shape":[1,4,128,256],"state_key":"model_input"},
        {"name":"timestep","dtype":"float32","shape":[1],"state_key":"timestep"},
        {"name":"encoder_hidden_states","dtype":"float32","shape":[1,512,2048],"state_key":"conditioning"},
        {"name":"encoder_attention_mask","dtype":"float32","shape":[1,512],"state_key":"attention_mask"},
        {"name":"time_ids","dtype":"float32","shape":[1,2],"state_key":"time_ids"}],
      "outputs":[{"name":"noise_pred","dtype":"float32","shape":list(reference.shape),"state_key":"model_output"}]}
    Path(a.metadata).write_text(json.dumps(meta,indent=2)+"\n")
    print(f"converted {a.output}; max_abs={max_abs:.8g}")

if __name__=="__main__": main()
