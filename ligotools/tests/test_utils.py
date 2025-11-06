import numpy as np
from pathlib import Path
from scipy.io import wavfile

from ligotools.utils import whiten, write_wavfile, reqshift

def test_write_wavfile_roundtrip(tmp_path):
    fs = 4096
    t = np.arange(0, 1.0, 1.0/fs)
    x = 0.5*np.sin(2*np.pi*440*t)
    out = tmp_path / "tone.wav"
    write_wavfile(str(out), fs, x)
    assert out.exists()
    fs2, y = wavfile.read(str(out))
    assert fs2 == fs
    assert len(y) == len(x)

def test_reqshift_shifts_frequency():
    fs = 4096
    t = np.arange(0, 1.0, 1.0/fs)
    f0, fshift = 440.0, 120.0
    x = np.sin(2*np.pi*f0*t)
    y = reqshift(x, fshift=fshift, sample_rate=fs)
    freqs = np.fft.rfftfreq(len(x), 1.0/fs)
    fX = freqs[np.argmax(np.abs(np.fft.rfft(x)))]
    fY = freqs[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs((fY - fX) - fshift) < 5.0  # within 5 Hz

def test_whiten_shape_and_finite():
    fs = 4096
    t = np.arange(0, 1.0, 1.0/fs)
    x = np.sin(2*np.pi*100*t) + 0.1*np.random.RandomState(0).normal(size=len(t))
    freqs = np.fft.rfftfreq(len(x), 1.0/fs)
    psd_flat = np.ones_like(freqs)
    interp_psd = lambda f: np.interp(f, freqs, psd_flat, left=1.0, right=1.0)
    w = whiten(x, interp_psd, dt=1.0/fs)
    assert len(w) == len(x)
    assert np.isfinite(w).all()
