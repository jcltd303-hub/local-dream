from export_contract import contract
from validate_metadata import validate

def test_official_contract_validates():
    m=contract()
    validate(m)
    u=m["components"]["unet"]
    assert u["inputs"][0]["shape"] == [1,4,128,256]
    assert u["inputs"][2]["shape"] == [1,-1,2048]
    assert u["inputs"][4]["shape"] == [1,2]

def test_latent_width_doubles_only_unet_spatial_input():
    m=contract(96,80)
    assert m["components"]["unet"]["inputs"][0]["shape"] == [1,4,96,160]
    assert m["components"]["vae_decoder"]["inputs"][0]["shape"] == [1,4,96,80]
