import numpy as np, pytest
from parity import check

def test_accepts_close_conversion():
    r=check(np.array([1.,2.]),np.array([1.0001,1.9999]))
    assert r["max_abs"] < 0.001

def test_rejects_shape_drift():
    with pytest.raises(ValueError,match="shape mismatch"):
        check(np.zeros((1,2)),np.zeros((2,1)))

def test_rejects_numerical_drift():
    with pytest.raises(ValueError,match="numerical parity failed"):
        check(np.array([0.]),np.array([1.]))
