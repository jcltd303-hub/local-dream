#!/usr/bin/env python3
"""Assemble per-component inspected/conversion metadata into one ABI input file."""
import argparse,json
from pathlib import Path
CORE=("unet","vae_encoder","vae_decoder")
def main():
 p=argparse.ArgumentParser();p.add_argument("--dir",required=True);p.add_argument("--output",required=True)
 p.add_argument("--multimodal-conditioning",action="store_true");a=p.parse_args()
 d=Path(a.dir); comps={}
 names=CORE if a.multimodal_conditioning else CORE+("text_encoder",)
 for n in names:
  f=d/f"{n}.json"
  if not f.is_file(): raise FileNotFoundError(f)
  x=json.loads(f.read_text()); comps[n]={"inputs":x["inputs"],"outputs":x["outputs"]}
 Path(a.output).write_text(json.dumps({"components":comps},indent=2)+"\n")
if __name__=="__main__": main()
