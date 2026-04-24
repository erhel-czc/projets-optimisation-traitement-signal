from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from scipy.signal import resample
from scipy.signal.windows import hann
from scipy.linalg import solve_toeplitz, toeplitz

# -----------------------------------------------------------------------------
# Block decomposition
# -----------------------------------------------------------------------------


def blocks_decomposition(x, w, R=0.5):
    """
    Performs the windowing of the signal

    Parameters
    ----------

    x: numpy array
      single channel signal
    w: numpy array
      window
    R: float (default: 0.5)
      overlapping between subsequent windows

    Return
    ------

    out: (blocks, windowed_blocks)
      block decomposition of the signal:
      - blocks is a list of the audio segments before the windowing
      - windowed_blocks is a list the audio segments after windowing
    """

    block_size = len(w)  # Taille de la fenêtre
    # Décalage entre les fenêtres (en fonction du taux de recouvrement R)
    hop_size = round(block_size * (1 - R))

    # Quick edge-effect mitigation: add zero-padding on both sides.
    pad = int(block_size * (1 - R))
    x = np.pad(x, pad, mode='constant')

    blocks = []
    windowed_blocks = []

    start = 0
    N = len(x)

    # Build overlapping blocks and zero-pad the last one if needed.
    while start < N:
        block = np.zeros(block_size)
        end = min(start + block_size, N)
        block[:end - start] = x[start:end]

        blocks.append(block)
        windowed_blocks.append(block * w)

        start += hop_size

    return np.array(blocks), np.array(windowed_blocks)


def blocks_reconstruction(blocks, w, signal_size, R=0.5):
    """
    Reconstruct a signal from overlapping blocks

    Parameters
    ----------

    blocks: numpy array
      signal segments. blocks[i,:] contains the i-th windowed
      segment of the speech signal
    w: numpy array
      window
    signal_size: int
      size of the original signal
    R: float (default: 0.5)
      overlapping between subsequent windows

    Return
    ------

    out: numpy array
      reconstructed signal
    """

    block_size = len(w)
    pad = int(block_size * (1 - R))
    hop_size = round(block_size * (1 - R))
    out = np.zeros(signal_size + block_size + pad)
    normalization = np.zeros(signal_size + block_size + pad)

    for i, block in enumerate(blocks):
        reconstructed_block = block * w
        start = i * hop_size
        end = min(start + block_size, out.size)
        size = end - start

        out[start:end] += reconstructed_block[:size]
        normalization[start:end] += (w**2)[:size]

    normalization = np.where(normalization < 1e-8, 1.0, normalization)
    out /= normalization

    return out[pad:pad + signal_size]
    # return out[:signal_size]

# -----------------------------------------------------------------------------
# Linear Predictive coding
# -----------------------------------------------------------------------------


def autocovariance(x, k):
    """
    Estimates the autocovariance C[k] of signal x

    Parameters
    ----------

    x: numpy array
      speech segment to be encoded
    k: int
      covariance index
    """

    n = len(x)
    return np.dot(x[:n - k], x[k:]) / n


def lpc_encode(x, p):
    """
    Linear predictive coding

    Predicts the coefficient of the linear filter used to describe the
    vocal track

    Parameters
    ----------

    x: numpy array
      segment of the speech signal
    p: int
      number of coefficients in the filter

    Returns
    -------

    out: tuple (coef, e, g)
      coefs: numpy array
        filter coefficients
      prediction: numpy array
        lpc prediction
    """

    N = len(x)

    r = np.array([autocovariance(x, k) for k in range(p + 1)])

    try:
        alpha = solve_toeplitz(r[:p], r[1:p + 1])
    except np.linalg.LinAlgError:
        # solution de secours en cas de matrice de Toeplitz mal conditionnée, solution trouvée par IA
        alpha = np.zeros(p)

    prediction = np.zeros(N)
    for i in range(p, N):
        prediction[i] = np.dot(alpha, x[i-p:i][::-1])

    return alpha, prediction


def lpc_decode(coefs, source):
    """
    Synthesizes a speech segment using the LPC filter and an excitation source

    Parameters
    ----------

    coefs: numpy array
      filter coefficients

    source: numpy array
      excitation signal

    Returns
    -------

    out: numpy array
      synthesized segment
    """
    p = len(coefs)
    N = len(source)

    out = np.zeros(N)
    for i in range(p, N):
        out[i] = np.dot(coefs, out[i-p:i][::-1]) + source[i]

    return out


def estimate_pitch(signal, sample_rate, min_freq=50, max_freq=200, threshold=1):
    """
    Estimate the pitch of an audio segment using the autocorrelation method and 
    indicate whether or not it is a voiced signal

    Parameters
    ----------

    signal: array-like
      audio segment
    sample_rate: int
      sample rate of the audio signal
    min_freq: int
      minimum frequency to consider (default 50 Hz)
    max_freq: int
      maximum frequency to consider (default 200 Hz)
    threshold: float
      threshold used to determine whether or not the audio segment is voiced

    Returns
    -------

    voiced: boolean
      Indicates if the signal is voiced (True) or not
    pitch: float
      estimated pitch (in s)
    """

    # A COMPLETER
    return 0, 0
