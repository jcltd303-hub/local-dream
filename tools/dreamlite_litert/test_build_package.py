import json,subprocess,sys
from pathlib import Path
from test_emit_abi import config
from export_contract import contract

def test_package_builder(tmp_path):
 c=tmp_path/"components";c.mkdir()
 names={"unet":"dreamlite_unet.tflite","vae_encoder":"dreamlite_vae_encoder.tflite",
        "vae_decoder":"dreamlite_vae_decoder.tflite","text_encoder":"dreamlite_text_encoder.tflite"}
 m=contract()
 for key,fn in names.items():
  (c/fn).write_bytes(b"x")
  (c/f"{key}.json").write_text(json.dumps(m["components"][key]))
 cp=tmp_path/"checkpoint.json";cp.write_text(json.dumps(config()))
 out=tmp_path/"package"
 subprocess.run([sys.executable,str(Path(__file__).with_name("build_package.py")),"--components",str(c),
   "--checkpoint-config",str(cp),"--output",str(out)],check=True)
 assert (out/"dreamlite_abi.json").is_file()
 cfg=json.loads((out/"config.json").read_text())
 assert cfg["runtime"]=="dreamlite_litert"
 assert cfg["dreamlite_multimodal_conditioning"] is False

def test_package_builder_marks_multimodal(tmp_path):
 c=tmp_path/"components";c.mkdir();m=contract()
 names={"unet":"dreamlite_unet.tflite","vae_encoder":"dreamlite_vae_encoder.tflite","vae_decoder":"dreamlite_vae_decoder.tflite","text_encoder":"dreamlite_text_encoder.tflite"}
 for key,fn in names.items():
  (c/fn).write_bytes(b"x");(c/f"{key}.json").write_text(json.dumps(m["components"][key]))
 cp=tmp_path/"checkpoint.json";cp.write_text(json.dumps(config()));out=tmp_path/"package"
 subprocess.run([sys.executable,str(Path(__file__).with_name("build_package.py")),"--components",str(c),"--checkpoint-config",str(cp),"--output",str(out),"--multimodal-conditioning"],check=True)
 assert json.loads((out/"config.json").read_text())["dreamlite_multimodal_conditioning"] is True
