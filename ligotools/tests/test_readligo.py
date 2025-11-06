import pytest
import numpy as np
from pathlib import Path

try:
    from ligotools.readligo import loaddata
except Exception:
    loaddata = None

DATA_DIR = Path("data")

def any_hdf5():
    if not DATA_DIR.exists():
        return None
    return next(DATA_DIR.glob("*.hdf5"), None)

@pytest.mark.skipif(loaddata is None, reason="loaddata not found in ligotools.readligo")
def test_loaddata_runs_and_returns_arrays():
    """
    Basic smoke test:
    Ensure loaddata loads a real .hdf5 file and returns arrays of equal length.
    """
    f = any_hdf5()
    if f is None:
        pytest.skip("No .hdf5 files found in data/")
    strain, time, dq = loaddata(str(f))
    assert len(strain) > 0
    assert len(strain) == len(time)


@pytest.mark.skipif(loaddata is None, reason="loaddata not found in ligotools.readligo")
def test_loaddata_sampling_rate_from_time():
    """Estimate fs from time spacing and check it's ~4096 Hz."""
    f = any_hdf5()
    if f is None:
        pytest.skip("No .hdf5 files found in data/")
    _, time, _ = loaddata(str(f))
    dt = np.diff(time)
    # all deltas positive and near constant
    assert np.all(dt > 0)
    fs_est = 1.0 / np.median(dt)
    assert abs(fs_est - 4096.0) < 1e-1  # allow tiny float tolerance