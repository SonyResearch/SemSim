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

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

loss_fn = LPIPS(net='squeeze')  #squeeze

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

    positive_item_ori = 0
    positive_item_rec = 0
    pm_results = []
    pm_results_positive_item = []
    for name in image_names:
        # Load original and reconstructed images
        ori_image = cv2.imread(os.path.join(ori_dir, name + "_ori.png"))
        rec_image = cv2.imread(os.path.join(rec_dir, name + "_rec.png"))

        # Calculate PSNR, SSIM, and MSE
        psnr_score = psnr(ori_image, rec_image)
        # ssim_score = ssim(ori_image, rec_image, multichannel=True) # different version
        ssim_score = ssim(ori_image, rec_image, channel_axis=-1)
        mse_score = mse(ori_image/255, rec_image/255)
        lpips_score = calculate_lpips(os.path.join(ori_dir, name + "_ori.png"), os.path.join(rec_dir, name + "_rec.png"))

        ori_ps = np.loadtxt(os.path.join(ori_dir+'_ps', name + ".txt"))
        rec_ps = np.loadtxt(os.path.join(rec_dir+'_ps', name + ".txt"))

        ori_model_rec = np.loadtxt(os.path.join(ori_dir + '_txt', name + "_model_rec.txt"))
        if ori_model_rec[0] == 1:
            positive_item_ori = positive_item_ori + 1
            pm_results_positive_item.append([name, psnr_score, ssim_score, mse_score, lpips_score, ori_ps*rec_ps])
            rec_model_rec = np.loadtxt(os.path.join(rec_dir + '_txt', name + "_model_rec.txt"))
            if rec_model_rec[0] ==1:
                positive_item_rec = positive_item_rec + 1

        # Add results to list
        pm_results.append([name, psnr_score, ssim_score, mse_score, lpips_score, ori_ps*rec_ps])
    return pm_results, pm_results_positive_item, positive_item_rec, positive_item_ori


if __name__ == '__main__':

    root_dir = 'benchmark/images/cifar100/'
    data_dir_raw= 'ResNet20-4_200_inversed_aug_auglist__rlabel_False_ori'
    with open('metrics/folder_names_cifar.txt', 'r') as f:
        lines = f.read().splitlines()


    for i in range(len(lines)):
        data_dir = lines[i]
        print(data_dir)

        path_ori= root_dir + data_dir_raw
        # path_ori= root_dir + data_dir +'_ori'
        path_rec = root_dir + data_dir

        metric_path = path_rec + '/metric_pixel_ssim.npy'
        metric_path_positive = path_rec + '/metric_pixel_positive.npy'
        if os.path.exists(metric_path):
            results_arr = np.load(metric_path, allow_pickle=True)
            results_positive_arr = np.load(metric_path_positive, allow_pickle=True)
        else:
            results, results_positive, positive_item_rec, positive_item_ori= calculate_metrics(path_ori, path_rec)
            results_arr = np.array(results, dtype=object)
            results_positive_arr = np.array(results_positive, dtype=object)
            np.save(os.path.join(path_rec, "metric_pixel.npy"), results_arr)
            np.save(os.path.join(path_rec, "metric_pixel_positive.npy"), results_positive_arr)
            acc = positive_item_rec/positive_item_ori


        mean_psnr = np.mean(results_arr[:,1])
        mean_ssim = np.mean(results_arr[:,2])
        mean_mse = np.mean(results_arr[:,3])
        mean_lpips_score = np.mean(results_arr[:,4])
        mean_ps = np.mean(results_arr[:,5])

        mean_ps_positive = np.mean(results_positive_arr[:,5])

        print('mean_psnr: {:.4f}; mean_ssim: {:.4f}; mean_mse: {:.4f}; mean_ps: {:.4f}'
              .format(mean_psnr, mean_ssim, mean_mse, mean_ps))

        print('mean_lpips_score: {:.4f}'
              .format(mean_lpips_score))


        print('mean_ps_positive: {:.4f}'
              .format(mean_ps_positive))


        print('acc: {:.4f}'
              .format(acc))



        print('--------------------------------------------------------')
