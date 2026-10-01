import copy, pytest
from test_emit_abi import metadata
from validate_metadata import validate

def test_accepts_official_mobile_contract():
    validate(metadata())

def test_rejects_wrong_text_width():
    m=metadata()
    for t in m["components"]["unet"]["inputs"]:
        if t["state_key"]=="conditioning": t["shape"]=[1,77,1024]
    with pytest.raises(ValueError,match="2048"): validate(m)

def test_rejects_non_diptych_width():
    m=metadata()
    for t in m["components"]["unet"]["inputs"]:
        if t["state_key"]=="model_input": t["shape"]=[1,4,128,255]
    m["components"]["unet"]["outputs"][0]["shape"]=[1,4,128,255]
    with pytest.raises(ValueError,match="two equal spatial halves"): validate(m)


def test_qwen_backend_accepts_no_text_component():
    m=metadata()
    del m["components"]["text_encoder"]
    validate(m,"qwen3_vl_gguf")
