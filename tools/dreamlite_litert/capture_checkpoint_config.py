#!/usr/bin/env python3
"""Capture scheduler/VAE constants from the actual DreamLite checkpoint."""
import argparse,json
from pathlib import Path

def val(config,name,default=None,required=True):
    v=getattr(config,name,default)
    if required and v is None: raise ValueError(f"checkpoint config missing {name}")
    return v

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    from dreamlite import DreamLiteMobilePipeline
    pipe=DreamLiteMobilePipeline.from_pretrained(a.model,dtype=None)
    s=pipe.scheduler.config; v=pipe.vae.config
    out={"scheduler":{
      "num_train_timesteps":int(val(s,"num_train_timesteps")),
      "use_dynamic_shifting":bool(val(s,"use_dynamic_shifting")),
      "time_shift_type":str(val(s,"time_shift_type")),
      "base_image_seq_len":int(val(s,"base_image_seq_len")),
      "max_image_seq_len":int(val(s,"max_image_seq_len")),
      "base_shift":float(val(s,"base_shift")),
      "max_shift":float(val(s,"max_shift")),
    },"vae":{
      "scaling_factor":float(val(v,"scaling_factor")),
      "shift_factor":float(val(v,"shift_factor",0.0,False) or 0.0),
    }}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
