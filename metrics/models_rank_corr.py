import os
import cv2
import numpy as np

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import torch
import torchvision.transforms as transforms
from lpips import LPIPS
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import kendalltau

from sklearn.linear_model import LinearRegression
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import numpy, pylab, scipy
# import seaborn as sns
# colors = sns.color_palette("Paired")

plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def confband(xd, yd, a, b, conf=0.95, x=None):
    """
   Calculates the confidence band of the linear regression model at the desired confidence
   level, using analytical methods. The 2sigma confidence interval is 95% sure to contain
   the best-fit regression line. This is not the same as saying it will contain 95% of
   the data points.

   Arguments:

   - conf: desired confidence level, by default 0.95 (2 sigma)
   - xd,yd: data arrays
   - a,b: linear fit parameters as in y=ax+b
   - x: (optional) array with x values to calculate the confidence band. If none is provided, will by default generate 100 points in the original x-range of the data.

   Returns:
   Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands
   corresponding to the [input] x array.

   Usage:

   >>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
   calculates the confidence bands for the given input arrays

   >>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
   plots a shaded area containing the confidence band

   References:
   1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
   2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm

   v1 Dec. 2011
   v2 Jun. 2012: corrected bug in computing dy
    """
    alpha = 1. - conf  # significance
    n = xd.size  # data sample size

    if x == None: x = np.linspace(xd.min(), xd.max(), 100)

    # Predicted values (best-fit model)
    y = a * x + b

    # Auxiliary definitions
    sd = scatterfit(xd, yd, a, b)  # Scatter of data about the model
    sxd = np.sum((xd - xd.mean()) ** 2)
    sx = (x - xd.mean()) ** 2  # array

    # Quantile of Student's t distribution for p=1-alpha/2
    q = scipy.stats.t.ppf(1. - alpha / 2., n - 2)

    # Confidence band
    dy = q * sd * np.sqrt(1. / n + sx / sxd)
    ucb = y + dy  # Upper confidence band
    lcb = y - dy  # Lower confidence band

    return lcb, ucb, x


def scatterfit(x, y, a=None, b=None):
    """
   Compute the mean deviation of the data about the linear model given if A,B
   (*y=ax+b*) provided as arguments. Otherwise, compute the mean deviation about
   the best-fit line.

   :param x,y: assumed to be Numpy arrays.
   :param a,b: scalars.
   :rtype: float sd with the mean deviation.
    """

    if a == None:
        # Performs linear regression
        a, b, r, p, err = scipy.stats.linregress(x, y)

    # Std. deviation of an individual measurement (Bevington, eq. 6.15)
    N = numpy.size(x)
    sd = 1. / (N - 2.) * np.sum((y - a * x - b) ** 2)
    sd = np.sqrt(sd)

    return sd


if __name__ == '__main__':

    data = np.loadtxt('metrics/data.txt')

    acc = data[:,0]
    psnr = data[:,1]
    ssim = data[:,2]
    mse = data[:,3]
    lpips = data[:,4]
    fid = data[:,5]
    s_fid = data[:,6]
    # l2 = data[:, 7]
    # cosin =  data[:, 8]

    ps = data[:, 10]
    classification = data[:, 12]

    human = data[:, 14]


    ours = np.loadtxt('metrics/semsim.txt')
    semsim = ours[:, 0]


    data1= 1 - human/600
    data2= psnr

   
    rho, pval = spearmanr(data1,data2)
    print("Spearman's rank correlation coefficient: {:.4f}".format(rho))
    print("p-value:", pval)

    p_corr, _ = pearsonr(data1,data2)
    print("Pearson rank correlation coefficient: {:.4f}".format(p_corr))

    t_corr, p_value = kendalltau(data1,data2)
    print("Kendall's Rank Correlation coefficient {:.4f}:".format(t_corr))
    print("p-value:", p_value)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.grid(True, color='gray', linewidth=2, linestyle="--", alpha=0.5)
    ax1 = plt.gca()
    ax1.patch.set_facecolor("w")

    ax1.patch.set_alpha(0.05)
    ax.set_axisbelow(True)


    # ------------------------------------------------------------------------------------
    slr = LinearRegression()
    slr.fit(data1.reshape(-1, 1), data2.reshape(-1, 1))
    data2_pred = slr.predict(data1.reshape(-1, 1))
    lr = plt.plot(data1.reshape(-1, 1), data2_pred.reshape(-1, 1), linestyle="-", linewidth=1.5, color="#589DBB")
    xdata = data1
    ydata = data2
    # Linear fit
    a, b, r, p, err = scipy.stats.linregress(xdata, ydata)
    # Generates arrays with the fit
    x = numpy.linspace(xdata.min(), xdata.max(), 100)
    y = a * x + b
    # Calculates the 2 sigma confidence band contours for the fit
    lcb, ucb, xcb = confband(xdata, ydata, a, b, conf=0.95)
    # Plots the confidence band as shaded area
    plt.fill_between(xcb, lcb, ucb, alpha=0.4, facecolor='#A4C7DB')
    # ------------------------------------------------------------------------------------


    sc = plt.scatter(data1, data2, color='#FDC9B4', s=800,  alpha= 0.9, edgecolor='red', linewidth=2)




    ax = plt.gca();
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)


    plt.xticks(size=35)
    plt.yticks(size=35)
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

    plt.show()
    plt.close()