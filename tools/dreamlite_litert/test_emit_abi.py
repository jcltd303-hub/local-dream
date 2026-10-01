import json, subprocess, sys, tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCRIPT=HERE/"emit_abi.py"

def t(name,key,shape=(1,)):
    return {"name":name,"dtype":"float32","shape":list(shape),"state_key":key}

def metadata():
    return {"components":{
      "unet":{"inputs":[t("sample","model_input",(1,4,128,256)),t("timestep","timestep"),
        t("encoder_hidden_states","conditioning",(1,77,2048)),t("encoder_attention_mask","attention_mask",(1,77)),
        t("time_ids","time_ids",(1,2))],"outputs":[t("noise_pred","model_output",(1,4,128,256))]},
      "vae_encoder":{"inputs":[t("image","reference_image")],"outputs":[t("latent","reference_latent")]},
      "vae_decoder":{"inputs":[t("latent","latent")],"outputs":[t("image","image")]},
      "text_encoder":{"inputs":[t("tokens","tokens")],"outputs":[t("hidden","conditioning")]},
    }}

def config():
    return {"scheduler":{"num_train_timesteps":1000,"use_dynamic_shifting":True,
      "time_shift_type":"exponential","base_image_seq_len":256,"max_image_seq_len":4096,
      "base_shift":0.5,"max_shift":1.16},"vae":{"scaling_factor":0.5,"shift_factor":0.1}}

def test_emit():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d); m=d/"m.json"; c=d/"c.json"; o=d/"dreamlite_abi.json"
        m.write_text(json.dumps(metadata())); c.write_text(json.dumps(config()))
        subprocess.run([sys.executable,str(SCRIPT),"--metadata",str(m),"--checkpoint-config",str(c),"--output",str(o)],check=True)
        out=json.loads(o.read_text())
        assert out["runtime"]=="dreamlite_litert"
        assert out["steps"]==4
        assert out["vae"]=={"scaling_factor":0.5,"shift_factor":0.1}
        assert out["components"]["unet"]["inputs"][0]["state_key"]=="model_input"

def test_rejects_missing_checkpoint_constant():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d); m=d/"m.json"; c=d/"c.json"; o=d/"x.json"
        bad=config(); del bad["vae"]["scaling_factor"]
        m.write_text(json.dumps(metadata())); c.write_text(json.dumps(bad))
        p=subprocess.run([sys.executable,str(SCRIPT),"--metadata",str(m),"--checkpoint-config",str(c),"--output",str(o)])
        assert p.returncode != 0

def test_emit_native_qwen_without_text_encoder():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d); m=d/"m.json"; c=d/"c.json"; o=d/"dreamlite_abi.json"
        meta=metadata(); del meta["components"]["text_encoder"]
        m.write_text(json.dumps(meta)); c.write_text(json.dumps(config()))
        subprocess.run([sys.executable,str(SCRIPT),"--metadata",str(m),"--checkpoint-config",str(c),
                        "--conditioning-backend","qwen3_vl_gguf","--output",str(o)],check=True)
        out=json.loads(o.read_text())
        assert out["conditioning_backend"]=="qwen3_vl_gguf"
        assert "text_encoder" not in out["components"]
