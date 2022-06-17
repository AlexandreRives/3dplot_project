from scipy import interpolate
from numpy.lib.stride_tricks import as_strided as strided
import numpy as np
import cv2
import pandas as pd

##############################
#       Kalman Filter        #
##############################

measurement = np.zeros((3,1),dtype=np.float32)
state = np.zeros((9,1),dtype=np.float32)
kalman = cv2.KalmanFilter(9,3,0)

def initKalman(x, y, z):
    measurement[0][0] = x
    measurement[1][0] = y
    measurement[2][0] = z
    kalman.statePre = np.zeros((9, 1), dtype=np.float32)
    kalman.statePre[0, 0] = x
    kalman.statePre[1, 0] = y
    kalman.statePre[2, 0] = z
    kalman.statePost = np.zeros((9, 1), dtype=np.float32)
    kalman.statePost[0, 0] = x
    kalman.statePost[1, 0] = y
    kalman.statePost[2, 0] = z
    kalman.measurementMatrix = cv2.setIdentity(kalman.measurementMatrix)
    kalman.processNoiseCov = cv2.setIdentity(kalman.processNoiseCov, 1)
    kalman.measurementNoiseCov = cv2.setIdentity(kalman.measurementNoiseCov, 10)
    kalman.errorCovPost = cv2.setIdentity(kalman.errorCovPost, 10)
    dt = 1 / 10
    v = dt
    a = 0.5 * (v ** 2)

    kalman.transitionMatrix = np.array([
        [1, 0, 0, v, 0, 0, a, 0, 0],
        [0, 1, 0, 0, v, 0, 0, a, 0],
        [0, 0, 1, 0, 0, v, 0, 0, a],
        [0, 0, 0, 1, 0, 0, v, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, v, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, v],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)


def kalmanPredict():
    prediction = kalman.predict()
    predictPr = [prediction[0, 0], prediction[1, 0], prediction[2, 0]]
    return predictPr


def kalmanCorrect(x, y, z):
    measurement[0, 0] = x
    measurement[1, 0] = y
    measurement[2, 0] = z
    estimated = kalman.correct(measurement)
    return [estimated[0, 0], estimated[1, 0], estimated[2, 0]]


##############################
#       Interpolation        #
##############################

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