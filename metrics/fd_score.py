#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.spatial.distance import cdist

import torch
from scipy import linalg
import imageio.v2 as imageio
from torch.nn.functional import adaptive_avg_pool2d, adaptive_max_pool2d
from scipy import misc
import random
import re
from scipy.special import softmax
from shutil import copyfile
import skimage


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
# from sklearn.cluster import KMeans
# import time

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from models.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('path', default='', type=str, nargs=2,
#                     help=('Path to the generated images or '
#                           'to .npz statistic files'))
parser.add_argument('--FD_model', type=str, default='inception', choices=['inception', 'posenet'],
                    help='model to calculate FD distance')
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='3', type=str,
                    help='GPU to use (leave blank for CPU only)')


def make_square(image, max_dim = 512):
    max_dim = max(np.shape(image)[0], np.shape(image)[1])
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image

def get_activations(opt, files, model, batch_size=50, dims=8192,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # if len(files) % batch_size != 0:
    #     print(('Warning: number of images is not a multiple of the '
    #            'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_remainder=  len(files) % batch_size

    print('\rnumber of batches is %d' % n_batches),
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs+n_remainder, dims))
    if n_remainder!=0:
        n_batches=n_batches+1
    for i in range(n_batches):
        # if verbose:
            # print('\rPropagating batch %d/%d' % (i + 1, n_batches),
            #       end='', flush=True)
        start = i * batch_size
        if n_remainder!=0 and i==n_batches-1:
          end = start + n_remainder
        else:
          end = start + batch_size

        # images = np.array(
        #       [skimage.transform.resize(imageio.imread(str(f)).astype(np.float32), size=[64, 64]).astype(np.float32)
        #        for f in files[start:end]])

        images = np.array([skimage.transform.resize(imageio.imread(str(f)).astype(np.float32), [64, 64]).astype(np.float32)
                           for f in files[start:end]])

        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()
        
        if opt.FD_model == 'inception':
            pred = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        if opt.FD_model == 'posenet':
            pred = model(batch)
            # print (np.shape (pred))
            pred = adaptive_max_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)
        # print('\rPropagating batch %d/%d' % (i + 1, n_batches))

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(opt, files, model, batch_size=50,
                                    dims=8192, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(opt, files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    #eigen_vals, eigen_vecs= np.linalg.eig(sigma)
    #sum_eigen_val=eigen_vals.sum().real
    sum_eigen_val = (sigma.diagonal()).sum()
    return act, mu, sigma, sum_eigen_val


def _compute_statistics_of_path(opt, path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        new_files=sorted(files, key=lambda p: int(str(p.stem).split('_')[0]))

        #random.shuffle(files)
        #files = files[:2000]

        feature, m, s, sum_eigen_val = calculate_activation_statistics(opt, new_files, model, batch_size,
                                               dims, cuda) 
    return feature, m, s, sum_eigen_val


def get_id_path_of_data (dataset_id, paths):
    img_paths = []
    dataset_ids = []
    person_ids = []
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    did = 0
    for sub_path in paths:
        sub_path = pathlib.Path(sub_path)
        files = list(sub_path.glob('*.jpg')) + list(sub_path.glob('*.png'))
        # files=glob.glob(osp.join(sub_path, '*.png'))+glob.glob(osp.join(sub_path, '*.jpg'))
        dataset_id_list = [dataset_id[did] for n in range(len(files))]
        dataset_ids.extend(dataset_id_list)
        img_paths.extend(files)
        did += 1
    dataset = []
    ii = 0
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(str(img_path)).groups())
        # if pid == -1: continue  # junk images are just ignored
        camid -= 1  # index starts from 0
        dataid = dataset_ids[ii]
        person_ids.append(pid)
        dataset.append((img_path, pid, dataid))
        ii = ii + 1

    return img_paths, person_ids, dataset_ids, dataset




def sort_data (paths):
    pattern = re.compile(r'([-\d]+)_([-\d]+)')

    paths_new= [None] * len(paths)

    for img_path in paths:
        pid, label = map(int, pattern.search(str(img_path.name)).groups())
        # print(pid)
        paths_new[pid-1] = img_path

    return paths_new


def calculate_fd_given_paths(paths, opt):
    """Calculates the FID of two paths"""
    # print(torch.cuda.is_available())
    cuda = True

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if opt.FD_model == 'inception':
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])

    if cuda:
        model.cuda()
    # 256 - 16


    feature_path = paths[0] + '/feature.npy'
    if os.path.exists(feature_path):
        feature1 = np.load(feature_path, allow_pickle=True)
        m1 = np.load(os.path.join(paths[0], "m.npy"), allow_pickle=True)
        s1 = np.load(os.path.join(paths[0], "s.npy"), allow_pickle=True)
        sum_eigen_val1 = np.load(os.path.join(paths[0], "sum_eigen_val.npy"), allow_pickle=True)
    else:
        feature1, m1, s1, sum_eigen_val1 = _compute_statistics_of_path(opt, paths[0], model, 1,
                                                                       dims, cuda)
        np.save(os.path.join(paths[0], "feature.npy"), feature1)
        np.save(os.path.join(paths[0], "m.npy"), m1)
        np.save(os.path.join(paths[0], "s.npy"), s1)
        np.save(os.path.join(paths[0], "sum_eigen_val.npy"), sum_eigen_val1)

    # npz_path = None
    # if not paths[0].endswith(".npz"):
    #     if not paths[0].endswith('/'):
    #         npz_path = paths[0] + ".npz"
    #     else:
    #         npz_path = paths[0][:-1] + ".npz"
    #     np.savez(npz_path, mu = m1, sigma = s1)

    feature_path = paths[1] + '/feature.npy'
    if os.path.exists(feature_path):
        feature2 = np.load(feature_path, allow_pickle=True)
        m2 = np.load(os.path.join(paths[1], "m.npy"), allow_pickle=True)
        s2 = np.load(os.path.join(paths[1], "s.npy"), allow_pickle=True)
        sum_eigen_val2 = np.load(os.path.join(paths[1], "sum_eigen_val.npy"), allow_pickle=True)
    else:
        feature2, m2, s2, sum_eigen_val2 = _compute_statistics_of_path(opt, paths[1], model, 1,
                                                                       dims, cuda)
        np.save(os.path.join(paths[1], "feature.npy"), feature2)
        np.save(os.path.join(paths[1], "m.npy"), m2)
        np.save(os.path.join(paths[1], "s.npy"), s2)
        np.save(os.path.join(paths[1], "sum_eigen_val.npy"), sum_eigen_val2)

    diff = feature1 - feature2
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    l2_distances = distances.mean()

    distances = cosine_distances(feature1, feature2)
    cos_distances = np.diag(distances).mean()


    fd_value = calculate_frechet_distance(m1, s1, m2, s2)

    single_fd_value=[]
    single_mean1 = []
    single_mean2 = []

    single_propensity_score=[]
    for i in range(len(feature1)):
        s_m1 = np.mean(feature1[i, :], axis=0)
        s_s1 = np.cov(feature1[i, :], rowvar=False)
        s_m2 = np.mean(feature2[i, :], axis=0)
        s_s2 = np.cov(feature2[i, :], rowvar=False)
        single_fd_value.append(calculate_frechet_distance(s_m1, s_s1, s_m2, s_s2))

        single_mean1.append(s_m1)
        single_mean2.append(s_m2)
        # single_propensity_score.append(calculate_propensity_score(feature1[i, :], feature2[i, :]))


    # mean = sum(single_fd_value) / len(single_fd_value)

    return single_fd_value, single_mean1, single_mean2, fd_value, l2_distances, cos_distances



def calculate_propensity_score(a, b):
    # Create labels for the control and treatment groups
    y_a = np.zeros(a.shape[0])
    y_b = np.ones(b.shape[0])

    # Concatenate the data
    X = np.concatenate((a, b))
    y = np.concatenate((y_a, y_b))

    # a b: a is _sets b is the rec_sets

    # Fit a logistic regression model
    model = LogisticRegression(random_state=0).fit(X, y)

    # Predict the propensity scores for the treatment group
    prop_scores = model.predict_proba(b)[:, 1]

    return prop_scores

if __name__ == '__main__':
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    root_dir = 'benchmark/images/cifar100/'
    data_dir_raw= 'ResNet20-4_200_inversed_aug_auglist__rlabel_False_ori'
    with open('metrics/folder_names_cifar.txt', 'r') as f:
        lines = f.read().splitlines()

    for i in range(0, 1):
        data_dir = lines[i]

        print(data_dir)

        path_ori= root_dir + data_dir_raw
        # path_ori= root_dir + data_dir +'_ori'
        path_rec = root_dir + data_dir

        args.path = [path_ori, path_rec]

        single_fd_value, single_mean1, single_mean2, fd_value, l2_distances, cos_distances  = calculate_fd_given_paths(args.path, args)


        sum_single_fd_value = sum(single_fd_value)
        print (fd_value)
        print(sum_single_fd_value)
        print(l2_distances)
        print(cos_distances)
