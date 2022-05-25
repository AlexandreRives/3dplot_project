from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided as strided

def low_pass_filter(cleanedList):

    #Filter requirements
    T = len(cleanedList)/200 # video time
    fps = 200 # frame per rate
    cutoff = 30 # desired cutoff frequency of the filter
    nyq = 0.5 * fps # Nyquist Frequency
    order = 2 # sin wave can be approx represented as quadratic
    normal_cutoff = cutoff / nyq

    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, cleanedList)
    return y


def mask_knans(a, x):
    a = np.asarray(a)
    k = a.size
    n = np.append(np.isnan(a), [False] * (x - 1))
    m = np.empty(k, np.bool8)
    m.fill(True)

    s = n.strides[0]
    i = np.where(strided(n, (k + 1 - x, x), (s, s)).all(1))[0][:, None]
    i = i + np.arange(x)
    i = pd.unique(i[i < k])

    m[i] = False

    return m


def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False)
    B = np.where(~mask_knans(A, 3), A, f(inds))
    return B