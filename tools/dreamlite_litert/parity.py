"""Shared numerical parity gate for converted DreamLite components."""
import numpy as np

def check(reference, converted, atol=5e-3, rtol=5e-3):
    a=np.asarray(reference); b=np.asarray(converted)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {b.shape} != {a.shape}")
    diff=np.abs(a-b)
    max_abs=float(diff.max()) if diff.size else 0.0
    if not np.allclose(a,b,atol=atol,rtol=rtol):
        raise ValueError(f"numerical parity failed: max_abs={max_abs}")
    return {"atol":atol,"rtol":rtol,"max_abs":max_abs}
