import numpy as np
from scipy.linalg import qr, svd, inv
import logging
import scipy.signal as scs
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from functools import wraps
import os, time
from sklearn.decomposition import PCA
import random

def canoncorr(X: np.array, Y: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA)
    line-by-line port from Matlab implementation of `canoncorr`
    X,Y: (samples/observations) x (features) matrix, for both: X.shape[0] >> X.shape[1]
    fullReturn: whether all outputs should be returned or just `r` be returned (not in Matlab)

    returns: A,B,r,U,V
    A,B: Canonical coefficients for X and Y
    U,V: Canonical scores for the variables X and Y
    r:   Canonical correlations

    Signature:
    A,B,r,U,V = canoncorr(X, Y)
    """
    n, p1 = X.shape
    p2 = Y.shape[1]
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Center the variables
    X = X - np.mean(X, 0);
    Y = Y - np.mean(Y, 0);

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1, T11, perm1 = qr(X, mode='economic', pivoting=True, check_finite=True)

    rankX = sum(np.abs(np.diagonal(T11)) > np.finfo(type((np.abs(T11[0, 0])))).eps * max([n, p1]));

    if rankX == 0:
        logging.error(f'stats:canoncorr:BadData = X')
    elif rankX < p1:
        logging.warning('stats:canoncorr:NotFullRank = X')
        Q1 = Q1[:, :rankX]
        T11 = T11[:rankX, :rankX]

    Q2, T22, perm2 = qr(Y, mode='economic', pivoting=True, check_finite=True)
    rankY = sum(np.abs(np.diagonal(T22)) > np.finfo(type((np.abs(T22[0, 0])))).eps * max([n, p2]));

    if rankY == 0:
        logging.error(f'stats:canoncorr:BadData = Y')
    elif rankY < p2:
        logging.warning('stats:canoncorr:NotFullRank = Y')
        Q2 = Q2[:, :rankY];
        T22 = T22[:rankY, :rankY];

    # Compute canonical coefficients and canonical correlations.  For rankX >
    # rankY, the economy-size version ignores the extra columns in L and rows
    # in D. For rankX < rankY, need to ignore extra columns in M and D
    # explicitly. Normalize A and B to give U and V unit variance.
    d = min(rankX, rankY);
    L, D, M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:, :d] * np.sqrt(n - 1);
    B = inv(T22) @ M[:, :d] * np.sqrt(n - 1);
    r = D[:d]
    # remove roundoff errs
    r[r >= 1] = 1
    r[r <= 0] = 0

    if not fullReturn:
        return r

    # Put coefficients back to their full size and their correct order
    A[perm1, :] = np.vstack((A, np.zeros((p1 - rankX, d))))
    B[perm2, :] = np.vstack((B, np.zeros((p2 - rankY, d))))

    # Compute the canonical variates
    U = X @ A
    V = Y @ B

    return A, B, r, U, V


def norm_gauss_window(bin_length, std):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_length : float
        binning length of the array we want to smooth in ms
    std : float
        standard deviation of the window
        use hw_to_std to calculate std based from half-width

    Returns
    -------
    win : 1D np.array
        Gaussian kernel with
            length: 10*std/bin_length
            mass normalized to 1
    """
    win = scs.gaussian(int(10*std/bin_length), std/bin_length)
    return win / np.sum(win)

def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))


def smooth_data(mat, dt=None, std=None, hw=None, win=None, backend='convolve1d'):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    dt : float
        length of the timesteps in seconds
    std : float (optional)
        standard deviation of the smoothing window
    hw : float (optional)
        half-width of the smoothing window
    win : 1D array-like (optional)
        smoothing window to convolve with
    backend: str, either 'convolve1d' or 'convolve'
        'convolve1d' (default) uses scipy.ndimage.convolve1d, which is faster in some cases
        'convolve'  uses scipy.signal.convolve, which may scale better for large arrays


    Returns
    -------
    np.array of the same size as mat
    """

    if win is None:
        assert dt is not None, "specify dt if not supplying window"

        if std is None:
            std = hw_to_std(hw)

        win = norm_gauss_window(dt, std)

    if mat.ndim != 1 and mat.ndim != 2:
        raise ValueError("mat has to be a 1D or 2D array")

    if backend == 'convolve1d':
        return convolve1d(mat, win, axis=0, output=np.float32, mode='reflect')
    elif backend == 'convolve':
        if mat.ndim == 1:
            return scs.convolve(mat, win, mode='same')
        elif mat.ndim == 2:
            return np.column_stack([scs.convolve(mat[:, i], win, mode='same') for i in range(mat.shape[1])])
    else:
        raise ValueError("backend has to either 'convolve1d' or 'convolve'")


def shaded_errorbar(ax: plt.axes, x: np.array, y: np.array = None, lineStat=np.mean, errorStat=np.std,
                    alpha=0.2, **props):
    """
    ax: axis to plot into
    x,y: data, solumns in y are collapsed to calculate the errorbar
    lineStat: a function to measure the midline, must accept an `axis` argument
    errorStat: a function to measure the  symmetric errorbars, must accept an `axis` argument
    most other keyword arguments will be passed to `plt.fill_between` and *some* to `plt.plot`
    """
    if y is None:
        y = x
        x = np.arange(y.shape[0])

    line = ax.plot(x, lineStat(y, axis=1))[0]

    shadeProps = props.copy()
    for key in props.keys():
        if key == "color" or key == "c":
            line.set_color(props[key])
        elif key == "linewidth" or key == 'lw':
            line.set_linewidth(props[key])
        elif key == "linestyle" or key == 'ls':
            line.set_linestyle(props[key])
        elif key == "marker":
            line.set_marker(props[key])
            shadeProps.pop(key, None)
        elif key == "markersize" or key == 'ms':
            line.set_markersize(props[key])
            shadeProps.pop(key, None)
        elif key == "label":
            line.set_label(props[key])
            shadeProps.pop(key, None)

    shadedY = errorStat(y, axis=1)
    shade = ax.fill_between(x, lineStat(y, axis=1) - shadedY, lineStat(y, axis=1) + shadedY,
                            alpha=alpha, **shadeProps)

    return line, shade

def report(func):
    "decorator to print the name and execution time of the function being executed."
    @wraps(func)
    def inner(*ar,**kar):
        print(f'Running: `{func.__name__}`...', end='\r')
        start = time.time()
        out = func(*ar,**kar)
        print(f'Executed: `{func.__name__}` in {time.time() - start:.1f}s')
        return out
    return inner

def smooth_average(mom_decom, window_len, step):
    '''
    Smooth the data using a moving average.
    data: matrix of voxels*samples
    window_len: window length
    step: step of the moving window
    '''
    window_len = window_len
    step = step
    mom_avg = []
    counting = 0
    while (counting <= mom_decom.shape[1] - 1):
        if counting == 0:
            mom_avg.append(np.mean(mom_decom[:, :counting + window_len // 2 + 1], 1))
            counting += step
        elif counting == mom_decom.shape[1] - 1:
            mom_avg.append(np.mean(mom_decom[:, counting - window_len // 2:], 1))
            break
        else:
            mom_avg.append(np.mean(mom_decom[:, counting - window_len // 2:counting + window_len // 2 + 1], 1))
            counting += step
    mom_avg = np.array(mom_avg).T

    return mom_avg

def down_sampling(data_list):
    voxel_list = [x.shape[0] for x in data_list]
    target_voxel = min(voxel_list)
    data_cut_list = []
    for i, voxel_num in enumerate(voxel_list):
        idx = random.sample(range(voxel_num),target_voxel)
        idx = np.sort(idx)
        data_cut_list.append(data_list[i][idx,:])

    return data_cut_list

def get_data_mat(data_list,n_components):
    model = PCA(n_components=n_components,svd_solver='full')
    rates = np.concatenate(data_list, axis=1)
    rates_model = model.fit(rates.T)
    data_pca = [rates_model.transform(s.T) for s in data_list]

    return data_pca, np.cumsum(rates_model.explained_variance_ratio_)

def eeg_bp_filter(data, fs, freqb='all', order=4):
    '''
    data: matrix of voxels*samples
    fs: sampling frequency
    freqb: frequency band for filtering, 'all' means no filtering, 'delta' [1 4], 'theta' [4 8], 'alpha' [8 12],
    'beta' [13 30], 'gamma' [30 40]
    order: order of filtering
    '''
    freqBand = {'delta': [1, 4],
                'theta': [4, 8],
                'alpha': [8, 12],
                'beta': [12, 30],
                'gamma': [30, 40]}
    if freqb not in freqBand:
        data_filter = data
    else:
        lf, hf = freqBand[freqb]
        wn1 = 2*lf/fs
        wn2 = 2*hf/fs
        b, a = scs.butter(order, [wn1, wn2], 'bandpass')
        data_filter = scs.filtfilt(b, a, data, axis=0)

    return data_filter

def divide_pair(data_list):
    '''
    Divide data in the input list into several pairs, each pair contains 2 data. All combinations will be delivered and
    no repeated

    data_list: List containing data to be divided
    data_pair: List containing divided pairs of data indices
    '''
    data_pair = []
    for j in range(len(data_list)):
        subj_ind = list(np.arange(len(data_list)))
        del subj_ind[j]
        for jj in range(len(subj_ind)):
            pair_temp = []
            pair_temp.append(j)
            pair_temp.append(subj_ind[jj])
            if list(np.sort(pair_temp)) not in data_pair:
                data_pair.append(pair_temp)

    return data_pair