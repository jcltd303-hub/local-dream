#!/usr/bin/env python3
"""Assemble per-component inspected/conversion metadata into one ABI input file."""
import argparse,json
from pathlib import Path
NAMES=("unet","vae_encoder","vae_decoder","text_encoder")
def main():
 p=argparse.ArgumentParser();p.add_argument("--dir",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 d=Path(a.dir); comps={}
 for n in NAMES:
  f=d/f"{n}.json"
  if not f.is_file(): raise FileNotFoundError(f)
  x=json.loads(f.read_text()); comps[n]={"inputs":x["inputs"],"outputs":x["outputs"]}
 Path(a.output).write_text(json.dumps({"components":comps},indent=2)+"\n")
if __name__=="__main__": main()
