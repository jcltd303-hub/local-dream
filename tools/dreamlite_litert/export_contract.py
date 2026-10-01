#!/usr/bin/env python3
"""Emit converter input metadata matching ByteVisionLab DreamLite Mobile export ABI.

This is framework-neutral: a Torch/ONNX/LiteRT conversion step may consume it,
then replace shapes with inspected converted-model shapes before emit_abi.py.
"""
import argparse, json
from pathlib import Path

def t(name, shape, state_key, dtype="float32"):
    return {"name":name,"dtype":dtype,"shape":shape,"state_key":state_key}

def contract(latent_height=128, latent_width=128, sequence_length=512):
    model_width=latent_width*2
    return {"components":{
      "unet":{"inputs":[
        t("sample",[1,4,latent_height,model_width],"model_input"),
        t("timestep",[1],"timestep"),
        t("encoder_hidden_states",[1,sequence_length,2048],"conditioning"),
        t("encoder_attention_mask",[1,sequence_length],"attention_mask"),
        t("time_ids",[1,2],"time_ids"),
      ],"outputs":[t("noise_pred",[1,4,latent_height,model_width],"model_output")]},
      "vae_encoder":{"inputs":[t("image",[1,3,-1,-1],"reference_image")],
                     "outputs":[t("latent",[1,4,latent_height,latent_width],"reference_latent")]},
      "vae_decoder":{"inputs":[t("latent",[1,4,latent_height,latent_width],"latent")],
                     "outputs":[t("image",[1,3,-1,-1],"image")]},
      "text_encoder":{"inputs":[t("tokens",[1,sequence_length],"tokens","int32")],
                      "outputs":[t("hidden",[1,sequence_length,2048],"conditioning"),
                                 t("attention_mask",[1,sequence_length],"attention_mask")]},
    }}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--latent-height",type=int,default=128)
    p.add_argument("--latent-width",type=int,default=128)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.latent_height<=0 or a.latent_width<=0: raise ValueError("latent dimensions must be positive")
    Path(a.output).write_text(json.dumps(contract(a.latent_height,a.latent_width),indent=2)+"\n")

if __name__=="__main__": main()
