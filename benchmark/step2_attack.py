from functools import partial
import os, sys

sys.path.insert(0, './')
import inversefed
import torch
import torchvision

seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random

random.seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import inversefed
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from inversefed.data.loss import LabelSmoothing
from inversefed.utils import Cutout
import torch.nn.functional as F
import policy
from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess, create_checkpoint_dir, create_save_dir
from transformers import ViTFeatureExtractor, ViTForImageClassification

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
# parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--aug_list', default='', required=False, type=str, help='Augmentation method.')
parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--resume', default=0, type=int, help='rlabel')
parser.add_argument('--semsim', default=False, type=bool, help='Semsim')

# 600 images used in attack
parser.add_argument('--rec_data_dir', default='benchmark/images/Cifar_ori/',
                    required=False, type=str, help='dir_of_rec_data to be test the acc')

parser.add_argument('--defense', default=None, type=str, help='Existing Defenses')
parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')
parser.add_argument('--dryrun', default=False, action='store_true', help='Debug mode')
parser.add_argument('--fix_ckpt', default=False, action='store_true', help='Use fix ckpt for attack')

opt = parser.parse_args()
num_images = 1

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative');
defs.epochs = opt.epochs

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']

config = create_config(opt)


def collate_fn(examples, label_key='fine_label'):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def reconstruct(idx, model, loss_fn, validloader_ori, validloader, mean_std, shape, label_key,aug_list):
    dm, ds = mean_std
    # prepare data
    ground_truth, labels, ground_truth_ori = [], [], []
    filenames = validloader.dataset.get_filename()
    if isinstance(model, ViTForImageClassification):
        # return tuple(logits,) instead of ModelOutput object
        model.forward = partial(model.forward, return_dict=False)
        while len(labels) < num_images:
            example = validloader.dataset[idx]
            example_ori = validloader_ori.dataset[idx]
            label = example[label_key]
            filename = filenames[idx]
            img_idx = int(filename.split("_")[0])
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(example)
                ground_truth_ori.append(example_ori)

        ground_truth = collate_fn(ground_truth, label_key=label_key)['pixel_values'].to(**setup)
        ground_truth_ori = collate_fn(ground_truth_ori, label_key=label_key)['pixel_values'].to(**setup)

    else:
        while len(labels) < num_images:
            img, label = validloader.dataset[idx]
            img_ori, _= validloader_ori.dataset[idx]
            filename = filenames[idx]
            img_idx = int(filename.split("_")[0])

            idx += 1
            if label not in labels:
                # print(label, type(label))
                if isinstance(label, torch.Tensor):
                    labels.append(label.to(device=setup['device']).unsqueeze(0))
                else:
                    labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))
                ground_truth_ori.append(img_ori.to(**setup))

        ground_truth = torch.stack(ground_truth)
        ground_truth_ori = torch.stack(ground_truth_ori)

    labels = torch.cat(labels)
    model.zero_grad()
    target_loss = loss_fn(model(ground_truth), labels)
    param_list = [param for param in model.parameters() if param.requires_grad]
    input_gradient = torch.autograd.grad(target_loss, param_list)

    # attack
    print('ground truth label is ', labels)
    # pass loss_fn that accepts tuple input
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images, loss_fn=loss_fn)

    if opt.rlabel:
        output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape)  # reconstruction label
    else:
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape,
                                                dryrun=opt.dryrun)  # specify label
        # output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape, dryrun=True) # specify label

    output_denormalized = output * ds + dm
    input_denormalized = ground_truth * ds + dm
    input_ori_denormalized = ground_truth_ori * ds + dm

    mean_loss = torch.mean((input_denormalized - output_denormalized) * (input_denormalized - output_denormalized))
    print("after optimization, the true mse loss {}".format(mean_loss))

    save_dir = create_save_dir(opt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/{}_{}_rec_{}.png'.format(save_dir, idx, int(labels.cpu()), aug_list))
    # torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/{}_{}_ori_{}.png'.format(save_dir, idx, int(labels.cpu()), aug_list))

    torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/{}_{}_rec.png'.format(save_dir, img_idx, int(labels.cpu())))
    torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/{}_{}_ori.png'.format(save_dir+'_ori', img_idx, int(labels.cpu())))

    test_mse = (output_denormalized.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
    if isinstance(model(output.detach()), tuple):
        feat_mse = (model(output.detach())[0] - model(ground_truth)[0]).pow(2).mean()
    else:
        feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()
    test_psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

    test_ori_mse = (output_denormalized.detach() - input_ori_denormalized).pow(2).mean().cpu().detach().numpy()
    if isinstance(model(output.detach()), tuple):
        feat_ori_mse = (model(output.detach())[0] - model(ground_truth_ori)[0]).pow(2).mean()
    else:
        feat_ori_mse = (model(output.detach()) - model(ground_truth_ori)).pow(2).mean()
    test_ori_psnr = inversefed.metrics.psnr(output_denormalized, input_ori_denormalized)


    return {'test_mse': test_mse,
            'feat_mse': feat_mse.detach(),
            # if not, the computation graph would store in list for each iteration, case OOM error. https://discuss.pytorch.org/t/memory-leak-when-appending-tensors-to-a-list/25937 If you store something from your model (for debugging purpose) and don’t need to calculate gradients with it anymore, I would recommend to call detach on it as it won’t have any effects if the tensor is already detached.
            'test_psnr': test_psnr,
            'test_ori_mse': test_ori_mse,
            'feat_ori_mse': feat_ori_mse.detach(),
            'test_ori_psnr': test_ori_psnr,
            }


def main():
    global trained_model
    print(opt)

    if opt.arch not in ['vit']:
        loss_fn, trainloader, validloader, recloader= preprocess(opt, defs, valid=True)
        _, _, validloader_ori, recloader_ori = preprocess(opt, defs, valid=False)
        model = create_model(opt)
        if opt.data == 'cifar100':
            # sxx
            dm = torch.as_tensor(inversefed.consts.cifar100_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.cifar100_std, **setup)[:, None, None]
            shape = (3, 32, 32)
        elif opt.data == 'FashionMinist':
            dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
            ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
            shape = (1, 32, 32)
        elif opt.data == 'Caltech101':
            dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
            shape = (3, 64, 64)
        elif opt.data == 'ImageNette':
            dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
            shape = (3, 112, 112)
        elif opt.data == 'Dog':
            dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
            shape = (3, 64, 64)
        elif opt.data == 'ImageNet':
            dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
            shape = (3, 224, 224)
        elif opt.data.startswith('CelebA'):
            dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]
            shape = (3, 64, 64)

        else:
            raise NotImplementedError
    else:
        loss_fn, trainloader, validloader, model, mean_std, scale_size = vit_preprocess(opt, defs,
                                                                                        valid=True)  # batch size rescale to 16
        _, _, validloader_ori, _, _, _ = vit_preprocess(opt, defs,
                                                                                        valid=True)  # batch size rescale to 16
        dm, ds = mean_std
        if opt.data == 'cifar100':
            dm = torch.as_tensor(dm, **setup)[:, None, None]
            ds = torch.as_tensor(ds, **setup)[:, None, None]
            shape = (3, scale_size, scale_size)
        elif opt.data == 'FashionMinist':
            dm = torch.Tensor(dm).view(1, 1, 1).cuda()
            ds = torch.Tensor(ds).view(1, 1, 1).cuda()
            shape = (1, scale_size, scale_size)

    label_key = 'fine_label' if opt.data == 'cifar100' else 'label'


    model.to(**setup)
    if opt.epochs == 0:
        trained_model = False

    if trained_model:
        checkpoint_dir = create_checkpoint_dir(opt)
        if 'normal' in checkpoint_dir:
            checkpoint_dir = checkpoint_dir.replace('normal', 'crop')
        filename = os.path.join(checkpoint_dir, f'{opt.arch}_{defs.epochs}.pth')
        # filename = os.path.join(checkpoint_dir, str(defs.epochs) + '.pth')

        if not os.path.exists(filename):
            filename = os.path.join(checkpoint_dir, str(defs.epochs - 1) + '.pth')

        print(filename)
        assert os.path.exists(filename)
        model.load_state_dict(torch.load(filename))

    if opt.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    model.eval()

    save_dir = create_save_dir(opt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'_ori')

    metric_list = list()
    # resume
    metric_path = save_dir + '/metric.npy'
    if os.path.exists(metric_path):
        metric_list = np.load(metric_path, allow_pickle=True).tolist()

    len_RI_data = len(metric_list)
    sample_list = [i for i in range(len_RI_data, 600)]

    if opt.arch == 'ResNet18_tv' and opt.data == 'ImageNet':
        valid_size = len(validloader.dataset)
        sample_array = np.linspace(0, valid_size, 100, endpoint=False, dtype=np.int32)
        sample_list = [int(i) for i in sample_array]


    mse_loss = 0
    for attack_id, idx in enumerate(sample_list):
        if idx < opt.resume:
            continue
        print('attach {}th in {}'.format(idx, opt.aug_list))
        metric = reconstruct(idx, model, loss_fn, recloader_ori, recloader, (dm, ds), shape, label_key, opt.aug_list)
        print(metric['test_psnr'])
        metric_list.append(metric)

        # save metric after each reconstruction
        np.save('{}/metric.npy'.format(save_dir), metric_list)

    psnr = []
    for b in metric_list:
        psnr.append(b['test_psnr'])
    print(sum(psnr) / len(psnr))


if __name__ == '__main__':
    main()





