#!/usr/bin/env python3
"""Convert DreamLite Mobile Tiny VAE encoder/decoder directly to LiteRT with parity gates."""
import argparse,json
from pathlib import Path
import numpy as np, torch, litert_torch
from parity import check

class Encoder(torch.nn.Module):
    def __init__(self,vae): super().__init__(); self.vae=vae
    def forward(self,image): return self.vae.encode(image,return_dict=False)[0]
class Decoder(torch.nn.Module):
    def __init__(self,vae): super().__init__(); self.vae=vae
    def forward(self,latent): return self.vae.decode(latent,return_dict=False)[0]

def convert(wrapper,args,out,meta,inputs,outputs,atol,rtol):
    wrapper.eval()
    with torch.no_grad(): ref=wrapper(*args).detach().cpu().numpy()
    edge=litert_torch.convert(wrapper,args)
    got=edge(*args)
    if isinstance(got,(tuple,list)): got=got[0]
    parity=check(ref,np.asarray(got),atol,rtol)
    Path(out).parent.mkdir(parents=True,exist_ok=True); edge.export(out)
    Path(meta).write_text(json.dumps({"inputs":inputs,"outputs":[{**outputs[0],"shape":list(ref.shape)}],"parity":parity},indent=2)+"\n")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--atol",type=float,default=5e-3);p.add_argument("--rtol",type=float,default=5e-3);a=p.parse_args()
    from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
    vae=AutoencoderTiny.from_pretrained(a.model,sub_folder="vae",torch_dtype=torch.float32).eval()
    d=Path(a.output_dir); g=torch.Generator().manual_seed(0)
    image=torch.randn(1,3,1024,1024,generator=g).clamp(-1,1)
    convert(Encoder(vae),(image,),d/"dreamlite_vae_encoder.tflite",d/"vae_encoder.json",
      [{"name":"image","dtype":"float32","shape":[1,3,1024,1024],"state_key":"reference_image"}],
      [{"name":"latent","dtype":"float32","shape":[1,4,128,128],"state_key":"reference_latent"}],a.atol,a.rtol)
    latent=torch.randn(1,4,128,128,generator=g)
    convert(Decoder(vae),(latent,),d/"dreamlite_vae_decoder.tflite",d/"vae_decoder.json",
      [{"name":"latent","dtype":"float32","shape":[1,4,128,128],"state_key":"latent"}],
      [{"name":"image","dtype":"float32","shape":[1,3,1024,1024],"state_key":"image"}],a.atol,a.rtol)

if __name__=="__main__": main()
