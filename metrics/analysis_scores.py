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
import shutil
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.linear_model import LinearRegression

import numpy as np
import numpy, pylab, scipy
# import seaborn as sns
# colors = sns.color_palette("Paired")
from scipy.stats import pearsonr
from scipy.stats import kendalltau
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


loss_fn = LPIPS(net='alex')

# test

def draw_correrlation_with_imgs(rec_dir, names, v1, v2):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(v1, v2, s=100)

    for x0, y0, name in zip(v1, v2, names):
        img_size = 0.32 * fig.dpi
        rec_img_dir = os.path.join(rec_dir, name + "_rec.png")
        rec_img = np.array(Image.open(rec_img_dir))
        ri = OffsetImage(rec_img, zoom=img_size / rec_img.shape[0])
        ri.image.axes = ax
        box = AnnotationBbox(ri, (x0, y0), xycoords='data', frameon=False)
        ax.add_artist(box)

    # set labels for axes
    # plt.xlabel('PSNR', fontsize=20)
    # plt.ylabel('SSIM', fontsize=20)
    plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    ax.spines['bottom'].set_linewidth(2)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()


def draw_correrlation(v1, v2):
    ##-- draw picture
    rho, pval = spearmanr(results_arr[:,1],results_arr[:,4])
    # print("psnr-lpips: pho: {:.4f}".format(rho))

    plt.figure(figsize=(10, 8))
    # create scatter plot
    plt.scatter(v1, v2, color='#6495ED', marker='o', edgecolor='black', s=150),

    # set labels for axes
    # plt.xlabel('PSNR')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax = plt.gca();  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(2)
    # ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.legend()

    label = ' p is ' + str(format(rho, '.4f'))  # 暂不支持中文
    loc = 'left'
    font_dict = {'fontsize': 28, \
                 'fontweight': 'semibold', \
                 'verticalalignment': 'baseline', \
                 'horizontalalignment': loc}
    # pad=3.5   待重新做，pad不能直接加进去
    plt.title(label, fontdict=font_dict, loc=loc)

    # display plot
    plt.show()
    plt.close()


def calculate_lpips(ori_img, rec_img):
        ori_image = Image.open(ori_img)
        rec_image = Image.open(rec_img)

        transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

        img1 = transform(ori_image)
        img2 = transform(rec_image)

        # Calculate the LPIPS distance between the two images
        lpips_value = loss_fn(img1, img2)

        return float(lpips_value)


def calculate_metrics(ori_dir, rec_dir):

    image_names = [img.split("_ori.png")[0] for img in os.listdir(ori_dir) if img.endswith("_ori.png")]
    image_names = sorted(image_names, key=lambda p: int(str(p).split('_')[0]))


    pm_results = []
    for name in image_names:
        # Load original and reconstructed images
        ori_image = cv2.imread(os.path.join(ori_dir, name + "_ori.png"))
        rec_image = cv2.imread(os.path.join(rec_dir, name + "_rec.png"))

        # Calculate PSNR, SSIM, and MSE
        psnr_score = psnr(ori_image, rec_image)
        ssim_score = ssim(ori_image, rec_image, multichannel=True)
        mse_score = mse(ori_image/255, rec_image/255)
        lpips_score = calculate_lpips(os.path.join(ori_dir, name + "_ori.png"), os.path.join(rec_dir, name + "_rec.png"))

        ori_ps = np.loadtxt(os.path.join(ori_dir+'_ps', name + ".txt"))
        rec_ps = np.loadtxt(os.path.join(rec_dir+'_ps', name + ".txt"))


        # Add results to list
        pm_results.append([name, psnr_score, ssim_score, mse_score, lpips_score, ori_ps, rec_ps])
    return pm_results



if __name__ == '__main__':
    metric = 'mse'
    # psnr  lpips pre_score  re_id_feature  class_feature reid_psnr_feature

    root_dir = 'benchmark/images/cifar100/'
    data_dir_raw= 'ResNet20-4_200_inversed_aug_auglist__rlabel_False_ori'
    with open('metrics/folder_names_cifar.txt', 'r') as f:
        lines = f.read().splitlines()
    save_dir = 'benchmark/images/anno/cifar-100-ana-img'+'_single'+ metric
    image_names = np.loadtxt('img_name_cifar.txt', dtype=str)

    # root_dir = 'benchmark/images/Caltech101/'
    # data_dir_raw= 'ResNet20-4_100_inversed_aug_auglist__rlabel_False_ori'
    # with open('metrics/folder_names_cal.txt', 'r') as f:
    #     lines = f.read().splitlines()
    # save_dir = 'benchmark/images/anno/cal_101_ana_img'+'_v2'+ metric
    # image_names = np.loadtxt('img_name_cal.txt', dtype=str)


    #
    # root_dir = 'benchmark/images/CelebA_Identity/'
    # data_dir_raw= 'ResNet20-4_100_inversed_cele_aug_auglist__rlabel_False_ori'
    # with open('metrics/folder_names_celeba.txt', 'r') as f:
    #     lines = f.read().splitlines()
    # save_dir = 'benchmark/images/anno/celeba_101_ana_img'+'_'+ metric
    # image_names = np.loadtxt('img_name_celeba.txt', dtype=str)



    # data_dir ='ConvNet_200_inversed_crop_auglist__rlabel_False_defense_prune_70'

    for i in range(len(lines)):
        data_dir = lines[i]
        # print(data_dir)

        path_ori= root_dir + data_dir_raw
        # path_ori= root_dir + data_dir +'_ori'
        path_rec = root_dir + data_dir

        metric_path = path_rec + '/metric_pixel.npy'
        if os.path.exists(metric_path):
            results_arr = np.load(metric_path, allow_pickle=True)
        else:
            results = calculate_metrics(path_ori, path_rec)
            results_arr = np.array(results, dtype=object)
            np.save(os.path.join(path_rec, "metric_pixel.npy"), results_arr)

        # mean_psnr = np.mean(results_arr[:,1])
        # mean_ssim = np.mean(results_arr[:,2])
        # mean_mse = np.mean(results_arr[:,3])
        # mean_lpips_score = np.mean(results_arr[:,4])
        # mean_ps = np.mean(results_arr[:,5]*results_arr[:,6])

        psnr = results_arr[:,1]
        ssim = results_arr[:,2]
        lpips_score = results_arr[:,4]
        # ori_score = results_arr[:, 5]
        # rec_score = results_arr[:, 6]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        image_names =results_arr[:,0]
        for id, name in enumerate(image_names):
            # Load original and reconstructed images

            if i == 0:
                ori_ps = np.loadtxt(os.path.join(path_ori + '_ps', name + ".txt"))
                ori_image = os.path.join(path_ori, name + "_ori.png")
                new_filename = name.split('_')[0] +'_{}_ori.png'.format(format(ori_ps, '.4f'))
                shutil.copy(ori_image, os.path.join(save_dir, new_filename))

            rec_ps = np.loadtxt(os.path.join(path_rec + '_ps', name + ".txt"))
            rec_image = os.path.join(path_rec, name + "_rec.png")
            new_filename = name.split('_')[0] +'_{}_{}_{}_{}_{}.png'.format(i, format(rec_ps, '.4f'),
                                                          format(psnr[id], '.4f'), format(ssim[id], '.4f'),
                                                          format(lpips_score[id], '.4f'),
                                                          )
            shutil.copy(rec_image, os.path.join(save_dir,new_filename))



    # rho, pval = spearmanr(results_arr[:,1],results_arr[:,2])
    # print("psnr-ssim: pho: {:.4f}".format(rho))


    # draw_correrlation_with_imgs(path_rec, results_arr[:,0], results_arr[:,1], results_arr[:,4])
    #
        # pm_results.append([name, psnr_score, ssim_score, mse_score, lpips_score, ori_ps*rec_ps])
        if i ==0:
        #     # plt.figure(figsize=(10, 8))
        #     # v1 = range(len(results_arr[:, 6]))
        #     # v2 = results_arr[:, 5]
        #     # plt.scatter(v1, v2, color='#6495ED', marker='o', s=150),
        #     # plt.show()
        #     # plt.close()
        #     # plt.figure(figsize=(10, 8))
        #     # v2 = results_arr[:, 6]
        #     # plt.scatter(v1, v2, color='#6495ED', marker='o', s=150),
        #     # plt.show()
        #     # plt.close()
        #     a= 1
        # else:
            # pm_results.append([name, psnr_score, ssim_score, mse_score, lpips_score, ori_ps*rec_ps])
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)
            # plt.figure(backgroundcolor='gary',    # 图表区的背景色
            #            edgecolor='black')    # 图表区的边框线颜色
            # ax.set_facecolor('#E5ECF6')
            # ax.set_facecolor((1.0, 0.47, 0.42))
            # plt.grid(True, color='w', linewidth=2)
            plt.grid(True, color='darkgray', linewidth=2, linestyle="--", alpha=0.9)
            ax1 = plt.gca()
            ax1.patch.set_facecolor("w")  # 设置 ax1 区域背景颜色

            ax1.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
            ax.set_axisbelow(True)


            data1 = np.array(results_arr[:, 2])
            data2 = np.array(results_arr[:, 4])
            data1_f = data1.astype(np.float64)
            data2_f = data2.astype(np.float64)

            rho, pval = spearmanr(data1, data2)
            print("Spearman's rank correlation coefficient: {:.4f}".format(rho))
            print("p-value:", pval)

            p_corr, _ = pearsonr(data1, data2)
            print("Pearson rank correlation coefficient: {:.4f}".format(p_corr))

            t_corr, p_value = kendalltau(data1, data2)
            print("Kendall's Rank Correlation coefficient {:.4f}:".format(t_corr))
            print("p-value:", p_value)


            # plt.scatter(data1_f, data2_f, color='#6495ED', marker='o', s=200, alpha=0.4),
            plt.scatter(data1_f, data2_f, color='#F11616', marker='o', s=200, alpha=0.2),


            slr = LinearRegression()
            slr.fit(data1_f.reshape(-1, 1), data2_f.reshape(-1, 1))
            data2_pred = slr.predict(data1_f.reshape(-1, 1))
            lr = plt.plot(data1_f.reshape(-1, 1), data2_pred.reshape(-1, 1), linestyle="-", linewidth=3, color="#F11616")
            xdata = data1_f
            ydata = data2_f
            # Linear fit
            a, b, r, p, err = scipy.stats.linregress(xdata, ydata)
            # Generates arrays with the fit
            x = numpy.linspace(xdata.min(), xdata.max(), 100)
            y = a * x + b
            # Calculates the 2 sigma confidence band contours for the fit
            lcb, ucb, xcb = confband(xdata, ydata, a, b, conf=0.95)
            # Plots the confidence band as shaded area
            plt.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor='#F11616')


            plt.xticks(size = 40) # human
            # plt.yticks(size = 40) # psnr
            # plt.yticks(np.arange(0, 0.1, 0.03), size = 40)  # ssim
            plt.yticks( size=40)  # psnr
            # plt.xticks(color='w')
            # plt.yticks(color='w')


            ax = plt.gca();  # 获得坐标轴的句柄
            ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)  # .set_color('none')
            # ax.spines['bottom'].set_color('gray')
            # ax.spines['left'].set_color('gray')
            # ax.spines['right'].set_color('gray')
            # ax.spines['top'].set_color('gray')



            plt.show()
            plt.close()