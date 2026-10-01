#!/usr/bin/env python3
"""Assemble a validated DreamLite LiteRT model package for local-dream."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
FILES={"unet":"dreamlite_unet.tflite","vae_encoder":"dreamlite_vae_encoder.tflite",
       "vae_decoder":"dreamlite_vae_decoder.tflite","text_encoder":"dreamlite_text_encoder.tflite"}
CORE=("unet","vae_encoder","vae_decoder")
def main():
 p=argparse.ArgumentParser();p.add_argument("--components",required=True);p.add_argument("--checkpoint-config",required=True)
 p.add_argument("--output",required=True);p.add_argument("--multimodal-conditioning",action="store_true")
 p.add_argument("--conditioning-llm");p.add_argument("--conditioning-vision");a=p.parse_args(); src=Path(a.components);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 required=CORE if a.multimodal_conditioning else CORE+("text_encoder",)
 for key in required:
  name=FILES[key];f=src/name
  if not f.is_file() or not f.stat().st_size: raise FileNotFoundError(f)
  shutil.copy2(f,out/name)
 if a.multimodal_conditioning:
  if not a.conditioning_llm or not a.conditioning_vision: raise ValueError("multimodal conditioning requires --conditioning-llm and --conditioning-vision")
  for source,name in [(a.conditioning_llm,"dreamlite_conditioning_llm.gguf"),(a.conditioning_vision,"dreamlite_conditioning_vision.gguf")]:
   f=Path(source)
   if not f.is_file() or not f.stat().st_size: raise FileNotFoundError(f)
   shutil.copy2(f,out/name)
 meta=out/"converted_metadata.json"
 assemble=[sys.executable,str(Path(__file__).with_name("assemble_metadata.py")),"--dir",str(src),"--output",str(meta)]
 if a.multimodal_conditioning: assemble.append("--multimodal-conditioning")
 subprocess.run(assemble,check=True)
 subprocess.run([sys.executable,str(Path(__file__).with_name("validate_metadata.py")),str(meta)],check=True)
 emit=[sys.executable,str(Path(__file__).with_name("emit_abi.py")),"--metadata",str(meta),
   "--checkpoint-config",a.checkpoint_config,"--output",str(out/"dreamlite_abi.json"),
   "--conditioning-backend","qwen3_vl_gguf" if a.multimodal_conditioning else "litert_text"]
 subprocess.run(emit,check=True)
 cfg={"runtime":"dreamlite_litert","default_steps":4,"dreamlite_unet":FILES["unet"],
      "dreamlite_vae_encoder":FILES["vae_encoder"],"dreamlite_vae_decoder":FILES["vae_decoder"],
      "dreamlite_multimodal_conditioning":bool(a.multimodal_conditioning)}
 if a.multimodal_conditioning:
  cfg["dreamlite_conditioning_llm"]="dreamlite_conditioning_llm.gguf"
  cfg["dreamlite_conditioning_vision"]="dreamlite_conditioning_vision.gguf"
 else:
  cfg["dreamlite_text_encoder"]=FILES["text_encoder"]
 (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
 print(out)
if __name__=="__main__":main()
