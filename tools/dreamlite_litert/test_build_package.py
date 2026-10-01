import json,subprocess,sys
from pathlib import Path
from test_emit_abi import config
from export_contract import contract

HERE=Path(__file__).resolve().parent
BUILDER=HERE/"build_package.py"

def write_components(root, include_text=True):
    root.mkdir()
    m=contract()
    names={"unet":"dreamlite_unet.tflite","vae_encoder":"dreamlite_vae_encoder.tflite",
           "vae_decoder":"dreamlite_vae_decoder.tflite"}
    if include_text: names["text_encoder"]="dreamlite_text_encoder.tflite"
    for key,fn in names.items():
        (root/fn).write_bytes(b"x")
        (root/f"{key}.json").write_text(json.dumps(m["components"][key]))

def test_package_builder(tmp_path):
    c=tmp_path/"components";write_components(c,True)
    cp=tmp_path/"checkpoint.json";cp.write_text(json.dumps(config()))
    out=tmp_path/"package"
    subprocess.run([sys.executable,str(BUILDER),"--components",str(c),
      "--checkpoint-config",str(cp),"--output",str(out)],check=True)
    abi=json.loads((out/"dreamlite_abi.json").read_text())
    cfg=json.loads((out/"config.json").read_text())
    assert abi["conditioning_backend"]=="litert_text"
    assert "text_encoder" in abi["components"]
    assert cfg["dreamlite_text_encoder"]=="dreamlite_text_encoder.tflite"
    assert cfg["dreamlite_multimodal_conditioning"] is False

def test_multimodal_package_omits_text_litert(tmp_path):
    c=tmp_path/"components";write_components(c,False)
    cp=tmp_path/"checkpoint.json";cp.write_text(json.dumps(config()));out=tmp_path/"package"
    llm=tmp_path/"llm.gguf";vision=tmp_path/"vision.gguf"
    llm.write_bytes(b"llm");vision.write_bytes(b"vision")
    subprocess.run([sys.executable,str(BUILDER),"--components",str(c),
      "--checkpoint-config",str(cp),"--output",str(out),"--multimodal-conditioning",
      "--conditioning-llm",str(llm),"--conditioning-vision",str(vision)],check=True)
    cfg=json.loads((out/"config.json").read_text())
    abi=json.loads((out/"dreamlite_abi.json").read_text())
    assert cfg["dreamlite_multimodal_conditioning"] is True
    assert "dreamlite_text_encoder" not in cfg
    assert not (out/"dreamlite_text_encoder.tflite").exists()
    assert abi["conditioning_backend"]=="qwen3_vl_gguf"
    assert "text_encoder" not in abi["components"]
    assert cfg["dreamlite_conditioning_llm"]=="dreamlite_conditioning_llm.gguf"
    assert cfg["dreamlite_conditioning_vision"]=="dreamlite_conditioning_vision.gguf"

def test_multimodal_package_requires_qwen_assets(tmp_path):
    c=tmp_path/"components";write_components(c,False)
    cp=tmp_path/"checkpoint.json";cp.write_text(json.dumps(config()));out=tmp_path/"package"
    p=subprocess.run([sys.executable,str(BUILDER),"--components",str(c),
      "--checkpoint-config",str(cp),"--output",str(out),"--multimodal-conditioning"])
    assert p.returncode != 0
