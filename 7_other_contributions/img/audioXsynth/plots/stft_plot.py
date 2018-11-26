#!/usr/bin/env python
# coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013
    Edited by L. Gabrielli 2017"""

import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
from matplotlib.mlab import find
import scikits.audiolab
import scipy.signal as SS
from scipy.fftpack import fft
from numpy.lib import stride_tricks
import librosa.display as lbd
import librosa.feature as lbf
import librosa

""" Parabolic interp from the web (common.py)"""
def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    xv = 1 / 2. * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4. * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)

""" short time fourier transform of audio signal """

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale)).astype('int')

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i + 1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i + 1]])]

    return newspec, freqs


""" plot spectrogram"""


def plotstft_chroma(audiopath, binsize=2 ** 10, plotpath=None, colormap="jet", pitch_vector=None):
    samples, samplerate = librosa.core.load(audiopath, sr=16000, mono=True, duration=14)

    # decimo per evitare robaccia inutile, SOLO PER FIGURA ESPERIMENTO 417!!
    print 'Se vuoi decimare (esperimento 417) qui dentro a plotstft_chroma lo puoi fare togliendo un commento qui sotto'
    #samples = SS.decimate(samples, 4)
    #samplerate = samplerate / 4
    #pitch_vector = pitch_vector / 2 # devo fare lo stesso con ivalori di pitch

    print 'Decimo per dimezzare altezza img'
    samples = SS.decimate(samples, 2)
    samplerate = samplerate / 2
    #pitch_vector = pitch_vector / 2 # devo fare lo stesso con ivalori di pitch

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(10, 3   ))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none", clim=(np.min(ims)+70,np.max(ims)))
    #plt.colorbar()

    plt.xlabel("Time [s]", fontsize=24)
    plt.ylabel("Frequency [Hz]", fontsize=24)
    plt.xlim([0, timebins - 1])
    plt.ylim([0, freqbins])

    plt.tick_params(axis='both', which='major', labelsize=20)
    xlocs = np.float32(np.linspace(0, timebins - 1, 4))
    plt.xticks(xlocs, ["%.0f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 6)))
    plt.yticks(ylocs, ["%.0f" % freq[i] for i in ylocs])

    if pitch_vector != None:
        plt.plot(pitch_vector, 'r')

    if plotpath:
        plt.savefig(plotpath, format='pdf', bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    ### CHROMA
    samples, samplerate = librosa.core.load(audiopath, sr=16000, mono=True, duration=14) # reload per annullare processing above
    chroma = lbf.chroma_cens(y=samples, sr=samplerate, hop_length=128)
    plt.figure(figsize=(10, 3))
    lbd.specshow(chroma, y_axis='chroma', x_axis='time', cmap='gray')
    plt.xlabel("Time [s]", fontsize=24)
    plt.ylabel("Pitch Class", fontsize=24, color='w')
    plt.tick_params(axis='y', which='major', labelsize=20, labelcolor='w', length=6, width=3)
    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.savefig("chromagram-"+plotpath, format='pdf', bbox_inches="tight")

""" extract pitch from a frame """
def f0_from_autocorr(signal, fs, fmax=700, fmin=200, prevResult=0.0):
    """Estimate frequency using autocorrelation
    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental
    Cons: Not as accurate, doesn't work for inharmonic things like musical
    instruments, this implementation has trouble with finding the true peak
    """
    # Calculate autocorrelation (same thing as convolution, but with one input
    # reversed in time), and throw away the negative lags
    signal -= np.mean(signal)  # Remove DC offset
    corr = SS.fftconvolve(signal, signal[::-1], mode='full')
    corr = corr[len(corr)/2:]

    # Find the first low point
    d = np.diff(corr)
    if max(d) > 0.0:
        start = find(d > 0)[0]

        # Find the next peak after the low point (other than 0 lag).  This bit is
        # not reliable for long signals, due to the desired peak occurring between
        # samples, and other peaks appearing higher.
        i_peak = np.argmax(corr[start:]) + start
        i_interp = parabolic(corr, i_peak)[0]
        result = fs / i_interp
        if result > fmax or result < fmin:
            return prevResult # for better tracking stability, allow propagating the past pitch predictions
        else:
            return result
    else:
        return 0.0

def pitch_tracker(input,fs):

    frs = 1024 * fs / 44100# framesize
    nframes = int(len(input)/frs)
    input = input[0:nframes*frs] # cut to make length multiple of frs
    input = np.transpose( np.reshape(input, (frs,len(input)/frs)) ) # array of frames)

    i = 0
    prev = 0.0
    pitchv = np.zeros(nframes+1)
    for row in input:
        pitchv[i] = f0_from_autocorr(row, fs, prevResult=prev)
        prev = pitchv[i]
        i += 1

    return pitchv

# if None:
#     pitchfile = "Vox.wav"
#     input, fs, enc = scikits.audiolab.wavread(pitchfile)
#     pive = pitch_tracker(input, fs)
#     plt.figure()
#     plt.plot(pive, 'r')
#
#     pitchfile = "reconstruction_417_batch_shuffle.wav"
#     input, fs, enc = scikits.audiolab.wavread(pitchfile)
#     pive = pitch_tracker(input, fs)
#     plt.plot(pive)
#     plt.show()
#




# SPECGRAM+CHROMAGRAM
pive = []
plotall = True
if plotall:
    plotstft_chroma("specgrams/Vox.wav", pitch_vector=pive, binsize=512, plotpath='specgram_Vox.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')
    plotstft_chroma("specgrams/output-reconstructed_406_P_ENV.wav", pitch_vector=pive, binsize=512, plotpath='specgram_406_P_ENV.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')
    plotstft_chroma("specgrams/vox_sweet_child_mfcc_match.wav", pitch_vector=pive, binsize=512, plotpath='specgram_mfcc-knn.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')
    plotstft_chroma("specgrams/vox_sweet_child_flatten.wav", pitch_vector=pive, binsize=512, plotpath='specgram_flatten.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')

    plotstft_chroma("specgrams/voceCaterina.wav", pitch_vector=pive, binsize=512, plotpath='specgram_Caterina.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')
    plotstft_chroma("specgrams/reconstruction_492_ENV.wav", pitch_vector=pive, binsize=512, plotpath='specgram_492_ENV.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')
    plotstft_chroma("specgrams/vox_caterina_mfcc_match.wav", pitch_vector=pive, binsize=512, plotpath='specgram_caterina-mfcc-knn.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')
    plotstft_chroma("specgrams/vox_caterina_flatten.wav", pitch_vector=pive, binsize=512, plotpath='specgram_caterina-flatten.pdf', colormap='gray')
    #plotstft("34-A3.wav", plotpath='specgram.pdf', colormap='gray')
else:
    plotstft_chroma("target_input_are_the_same/femalevoice.wav", pitch_vector=pive, binsize=512, plotpath='specgram_selfcodingVox.pdf', colormap='gray')
    plotstft_chroma("target_input_are_the_same/sweetchild-cut.wav", pitch_vector=pive, binsize=256,
                    plotpath='specgram_selfcodingChild.pdf', colormap='gray')
    plotstft_chroma("sweetchild-orig-cut22khz.wav", pitch_vector=pive, binsize=256,
                    plotpath='specgram_orig-Child.pdf', colormap='gray')
