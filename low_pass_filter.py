from scipy.signal import butter, filtfilt
import numpy as np
from numpy.lib.stride_tricks import as_strided as strided

def low_pass_filter(coord, fps, cutoff):
    """
    Returns filtered coordinates

    Parameters :
    --------------
    coord : numpy array
        list of coordinates of each dimension (x, y or z) to filter
    """

    #Filter requirements
    nyq = 0.5 * fps # Nyquist Frequency
    order = 2 # sin wave can be approx represented as quadratic
    normal_cutoff = cutoff / nyq

    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = np.zeros(coord.shape)*float('NaN')
    y[~np.isnan(coord)] = filtfilt(b, a, coord[~np.isnan(coord)])
    return y