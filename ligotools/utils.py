import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import mlab 

def whiten(strain, interp_psd, dt):
    """
    Whiten a strain time series using an interpolated one-sided PSD.
    Parameters
    ----------
    strain : 1D array
    interp_psd : callable
        Function f(freq) -> PSD(freq). e.g., scipy.interpolate.interp1d
    dt : float
        Sample spacing (seconds), e.g. 1/fs
    """
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    white_hf = hf / np.sqrt(interp_psd(freqs) / 2.0)
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(filename, fs, data):
    """
    Write scaled audio to WAV (int16). Safe when max(abs(data)) == 0.
    """
    peak = float(np.max(np.abs(data))) or 1.0
    d = np.int16((data / peak) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)

def reqshift(data, fshift=100.0, sample_rate=4096.0):
    """
    Frequency-shift a band-passed signal by fshift (Hz) using FFT roll.
    """
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(np.round(fshift / df))
    y = np.roll(x, nbins)
    y[:nbins] = 0
    z = np.fft.irfft(y, n=len(data))
    return z

def plot_asds_with_model(strain_H1, strain_L1, fs, eventname, plottype="png",
                         f_min=20.0, f_max=2000.0, outdir="figures"):
    """
    Recreates the 'ASD with model' plot from the tutorial cell that begins with
    '# -- To calculate the PSD of the data, choose an overlap and a window...'.
    Saves to figures/{eventname}_ASDs.{plottype}.

    Returns
    -------
    freqs : 1D array of frequencies
    psd_H1, psd_L1 : one-sided PSDs for H1/L1 (from mlab.psd)
    psd_smooth : callable interpolant for the smooth model (interp1d)
    """
    NFFT = int(4 * fs)
    Pxx_H1, freqs = mlab.psd(strain_H1, Fs=fs, NFFT=NFFT)
    Pxx_L1, _     = mlab.psd(strain_L1, Fs=fs, NFFT=NFFT)

    Pxx_model = (1.0e-22 * (18.0 / (0.1 + freqs))**2
                 + 0.7e-23 * 2
                 + ((freqs / 2000.0) * 4.0e-23)**2)
    Pxx_model = Pxx_model**2  # convert amplitude-like expression to PSD
    psd_smooth = interp1d(freqs, Pxx_model, bounds_error=False, fill_value=np.inf)

    # Plot ASDs
    plt.figure(figsize=(10, 8))
    plt.loglog(freqs, np.sqrt(Pxx_L1), 'g', label='L1 strain')
    plt.loglog(freqs, np.sqrt(Pxx_H1), 'r', label='H1 strain')
    plt.loglog(freqs, np.sqrt(Pxx_model), 'k', label='H1 strain, O1 smooth model')
    plt.axis([f_min, f_max, 1e-24, 1e-19])
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.ylabel('ASD (strain/rtHz)')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc='upper center')
    plt.title(f'Advanced LIGO strain data near {eventname}')
    out = f"{outdir}/{eventname}_ASDs.{plottype}"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return freqs, Pxx_H1, Pxx_L1, psd_smooth
