import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import mlab

# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

# function to keep the data within integer limits, and write to wavfile:
def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

# function that shifts frequency of a band-passed signal
def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z

def plot_match_results(
    time, timemax, SNR, det, eventname, plottype,
    strain_whitenbp, template_match,
    datafreq, template_fft, d_eff,
    freqs, data_psd
):
    """
    Recreates the three plots from the matched-filter cell:
      1) SNR(t) (full + zoom)
      2) whitened data vs template + residuals (zoomed)
      3) ASD + template in frequency

    Saves:
      figures/{eventname}_{det}_SNR.{plottype}
      figures/{eventname}_{det}_matchtime.{plottype}
      figures/{eventname}_{det}_matchfreq.{plottype}
    """
    # pick color and labels by detector
    pcolor = 'g' if det == 'L1' else 'r'

    # 1) SNR(t): full + zoom
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - timemax, SNR, pcolor, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.title(f"{det} matched filter SNR around event")

    plt.subplot(2, 1, 2)
    plt.plot(time - timemax, SNR, pcolor, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlim([-0.15, 0.05])
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.savefig(f"figures/{eventname}_{det}_SNR.{plottype}", bbox_inches="tight")
    plt.close()

    # 2) whitened data vs template + residuals
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, strain_whitenbp, pcolor, label=f"{det} whitened h(t)")
    plt.plot(time, template_match, "k", label="Template(t)")
    plt.ylim([-10, 10])
    plt.xlim([time.min(), time.min() + 0.20])  # ~[-0.15, +0.05] relative zoom
    plt.grid(True)
    plt.xlabel(f"Time (s)")
    plt.ylabel("whitened strain (noise stdev units)")
    plt.legend(loc="upper left")
    plt.title(f"{det} whitened data around event")

    plt.subplot(2, 1, 2)
    plt.plot(time, strain_whitenbp - template_match, pcolor, label=f"{det} resid")
    plt.ylim([-10, 10])
    plt.xlim([time.min(), time.min() + 0.20])
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("whitened strain (noise stdev units)")
    plt.legend(loc="upper left")
    plt.title(f"{det} Residual whitened data after subtracting template")
    plt.savefig(f"figures/{eventname}_{det}_matchtime.{plottype}", bbox_inches="tight")
    plt.close()

    # 3) ASD + template(f)*sqrt(f)
    plt.figure(figsize=(10, 6))
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, "k", label="template(f)*sqrt(f)")
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=f"{det} ASD")
    plt.xlim(20, datafreq.max())
    plt.ylim(1e-24, 1e-20)
    plt.grid(True)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("strain noise ASD (strain/rtHz), template h(f)*rt(f)")
    plt.legend(loc="upper left")
    plt.title(f"{det} ASD and template around event")
    plt.savefig(f"figures/{eventname}_{det}_matchfreq.{plottype}", bbox_inches="tight")
    plt.close()