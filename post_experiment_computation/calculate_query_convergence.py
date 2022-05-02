from typing import Dict
from matplotlib import pyplot as plot # type: ignore
from sklearn.neighbors import NearestNeighbors

import numpy as np
import os

folder = "./experiment_results/run13"

def load_from_path():
        walk_score_files(os.path.join(folder, "log_scores_over_time"))
        load_gt_file(os.path.join(folder, "log_gt_scores"))

def walk_score_files(path):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npy"):
                data = np.load(os.path.join(dirpath, f))
                load_score_data(data, algorithm="gt_scores")
                load_score_data(data, algorithm="actual")

knn = NearestNeighbors(n_neighbors=3)
base_data = None
best_score = None
def load_gt_file(path):
    global base_data

    for dirpath, dnames, fnames in os.walk(path):
        global best_score
        f: str
        for f in fnames:
            if f.endswith(".npy"):
                data = np.load(os.path.join(dirpath, f))
                base_data = data[0,...,0]
                scores = base_data[1,:,0]
                queries = base_data[0,:,:]
                knn.fit(queries, scores)
                best_score = np.max(scores)


def score_of_query(queries):
    indexes = knn.kneighbors(queries, return_distance=False)
    gt_scores = base_data[1,:,0]
    scores = gt_scores[indexes]
    return np.mean(scores, axis=1)


algorithm_data: Dict = {}
def load_score_data(data, algorithm):
    runs_data = algorithm_data.get(algorithm, [])
    runs_data.append(data)
    algorithm_data[algorithm] = runs_data


algorithm_gt_scores: Dict = {}
def extract_gt_scores():
    for item in algorithm_data.items():
        data = np.asarray(item[1])
        if item[0] == "gt_scores":
            queries = data[:,:,0,:,:,0]
            org_shape = queries.shape[0:-1]
            queries = queries.reshape((-1, 2))
            scores = score_of_query(queries)
            scores = scores.reshape(org_shape)
        else:
            scores = data[:,:,1,:,0,0]
        algorithm_gt_scores[item[0]] = scores

algorithm_means: Dict = {}
def calculate_means():
    for item in algorithm_gt_scores.items():
        scores = np.asarray(item[1])
        mean = np.nanmean(scores, axis=(0,2))
        algorithm_means[item[0]] = mean

algorithm_medians: Dict = {}
def calculate_medians():
    for item in algorithm_gt_scores.items():
        scores = np.asarray(item[1])
        median = np.nanmedian(scores, axis=(0,2))
        algorithm_medians[item[0]] = median

algorithm_vars: Dict = {}
def calculate_vars():
    for item in algorithm_gt_scores.items():
        scores = np.asarray(item[1])
        var = np.nanvar(scores, axis=(0,2))
        algorithm_vars[item[0]] = var

def plot_gt_score(fig_name = "gt_score"):

    for key in algorithm_data.keys():
        y = algorithm_means[key]
        x = np.asarray([i for i in range(y.shape[0])])
        y_err = algorithm_vars[key]
        plot.ylim(0,1.1)
        plot.errorbar(x, y, y_err, alpha=0.5, fmt=' ', label=f'{key}_var')

        print(best_score)
        x = np.asarray([i for i in range(y.shape[0])])
        y = np.asarray([best_score for i in range(y.shape[0])])
        plot.ylim(0,1.1)
        plot.plot(x, y, alpha=1, label=f'gt_max')
    
    for key in algorithm_data.keys():
        y = algorithm_means[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(0,1.1)
        plot.plot(x, y, alpha=1, label=f'{key}_mean')

    for key in algorithm_data.keys():
        y = algorithm_medians[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(0,1.1)
        plot.plot(x, y, alpha=1, label=f'{key}_median')

    plot.title(fig_name)
    plot.xlabel("learning iteration")
    plot.ylabel("gt-score")
    plot.figlegend()
    
    plot.savefig(f'{folder}/{fig_name}.png',dpi=500)
    plot.clf()

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
     r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
     The Savitzky-Golay filter removes high frequency noise from data.
     It has the advantage of preserving the original shape and
     features of the signal better than other types of filtering
     approaches, such as moving averages techniques.
     Parameters
     ----------
     y : array_like, shape (N,)
         the values of the time history of the signal.
     window_size : int
         the length of the window. Must be an odd integer number.
     order : int
         the order of the polynomial used in the filtering.
         Must be less then `window_size` - 1.
     deriv: int
         the order of the derivative to compute (default = 0 means only smoothing)
     Returns
     -------
     ys : ndarray, shape (N)
         the smoothed signal (or it's n-th derivative).
     Notes
     -----
     The Savitzky-Golay is a type of low-pass filter, particularly
     suited for smoothing noisy data. The main idea behind this
     approach is to make for each point a least-square fit with a
     polynomial of high order over a odd-sized window centered at
     the point.
     Examples
     --------
     t = np.linspace(-4, 4, 500)
     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
     ysg = savitzky_golay(y, window_size=31, order=4)
     import matplotlib.pyplot as plt
     plt.plot(t, y, label='Noisy signal')
     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
     plt.plot(t, ysg, 'r', label='Filtered signal')
     plt.legend()
     plt.show()
     References
     ----------
     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688  [Titel anhand dieser ISBN in Citavi-Projekt übernehmen] 
     """
     import numpy as np
     from math import factorial
     
     try:
         window_size = np.abs(np.int(window_size))
         order = np.abs(np.int(order))
     except ValueError as msg:
         raise ValueError("window_size and order have to be of type int")
     if window_size % 2 != 1 or window_size < 1:
         raise TypeError("window_size size must be a positive odd number")
     if window_size < order + 2:
         raise TypeError("window_size is too small for the polynomials order")
     order_range = range(order+1)
     half_window = (window_size -1) // 2
     # precompute coefficients
     b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
     m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
     # pad the signal at the extremes with
     # values taken from the signal itself
     firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
     lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
     y = np.concatenate((firstvals, y, lastvals))
     return np.convolve( m[::-1], y, mode='valid')

algorithm_mean_grads: Dict = {}
def calculate_mean_grads():
    for key in algorithm_data.keys():
        means = algorithm_means[key]
        grad = savitzky_golay(means,45,4,1)
        algorithm_mean_grads[key] = grad

def plot_learning_gradient(fig_name = "learning_gradient"):
    
    for key in algorithm_data.keys():
        y = algorithm_mean_grads[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(-0.01,0.01)
        plot.plot(x, y, label=f'{key}_grad')

    ax = plot.gca()
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plot.title(fig_name)
    plot.xlabel("learning iteration")
    plot.ylabel("score-gradient")
    plot.figlegend()
    
    plot.savefig(f'{folder}/{fig_name}.png',dpi=500)
    plot.clf()

gain_mean_p: Dict = {}
def calculate_p_mean_gain():
    for key in algorithm_data.keys():
        mean = algorithm_means[key]
        base_mean = algorithm_means[baseline]
        gain = (base_mean - mean) / base_mean
        gain_mean_p[key] = gain

gain_median_p: Dict = {}
def calculate_p_median_gain():
    for key in algorithm_data.keys():
        median = algorithm_medians[key]
        base_median = algorithm_medians[baseline]
        gain = (base_median - median) / base_median
        gain_median_p[key] = gain


def plot_p_gain(p_gain, fig_name = "p-gain"):
    for key in algorithm_data.keys():
        y = p_gain[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(-1,1)
        plot.plot(x, y, label=f'{key}')

    plot.title(fig_name)
    plot.xlabel("learning iteration")
    plot.ylabel("p-value gain against baseline")
    plot.figlegend()
    
    plot.savefig(f'{folder}/{fig_name}.png',dpi=500)
    plot.clf()

max_p = 0.25
mean_data_gain: Dict = {}
def calculate_mean_data_gain():
    for key in algorithm_data.keys():
        mean = algorithm_means[key]
        base_mean = algorithm_means[baseline]

        mean_data_gain[key] = calculate_data_gain(mean, base_mean)

median_data_gain: Dict = {}
def calculate_median_data_gain():
    for key in algorithm_data.keys():
        median = algorithm_medians[key]
        base_median = algorithm_medians[baseline]

        median_data_gain[key] = calculate_data_gain(median, base_median)

def calculate_data_gain(mean, base_mean):
        ps = np.linspace(max_p, 0)
        itters = []
        base_itters = []
        gains_p = []
        for i in range(ps.shape[0]):
            itter = np.where(mean <= ps[i])
            base_itter = np.where(base_mean<= ps[i])

            try:
                base_itter = base_itter[0][0]
            except IndexError:
                base_itter = base_mean.shape[0]

            try:
                itter = itter[0][0]

                itters.append(itter)
                base_itters.append(base_itter)
                gains_p.append(ps[i])

            except IndexError:
                ...

        itters = np.asarray(itters)
        base_itters = np.asarray(base_itters)
        gains_p = np.asarray(gains_p)

        gain = (base_itters - itters) / base_itters
        return (gains_p, gain)


def plot_data_gain(data_gain, fig_name = "data-gain"):
    for key in algorithm_data.keys():
        x, y = data_gain[key]
        plot.xlim(max_p,0)
        plot.ylim(-1,1)
        plot.plot(x, y, label=f'{key}')

    plot.title(fig_name)
    plot.xlabel("p-value")
    plot.ylabel("data gain against baseline")
    plot.figlegend()
    
    plot.savefig(f'{folder}/{fig_name}.png',dpi=500)
    plot.clf()




load_from_path()

extract_gt_scores()

calculate_means()
calculate_medians()
calculate_vars()

plot_gt_score()

calculate_mean_grads()

plot_learning_gradient()

# calculate_p_mean_gain()
# calculate_p_median_gain()

# plot_p_gain(gain_mean_p, "p-mean-gain")
# plot_p_gain(gain_median_p, "p-median-gain")

# calculate_mean_data_gain()
# calculate_median_data_gain()

# plot_data_gain(mean_data_gain, "mean-data-gain")
# plot_data_gain(median_data_gain, "median-data-gain")
