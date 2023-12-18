import os, sys

sys.path.insert(0, './')
import torch

seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
import torch.nn as nn
random.seed(seed)

import numpy as np
import inversefed
import argparse

import policy

from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess, create_checkpoint_dir

policies = policy.policies

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
# not use in attack
parser.add_argument('--rec_data_dir', default='/home/sunxx/project/ATSPrivacy/benchmark/images/Cifar_ori_600/data_used_generate_RI',
                    required=False, type=str, help='dir_of_rec_data to be test')

parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--aug_list', default='', required=False, type=str, help='Augmentation method.')

parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='remove label.')
parser.add_argument('--evaluate', default=False, type=bool, help='Evaluate')
parser.add_argument('--pretrained', default=False, type=bool, help='Pretrained')
parser.add_argument('--semsim', default=False, type=bool, help='semsim')
parser.add_argument('--semsim_psnr', default=False, type=bool, help='semsim')


parser.add_argument('--targte_data', default='cifar100', required=False, type=str, help='targte_dataset.')

parser.add_argument('--defense', default=None, type=str, help='Existing Defenses')
parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')

opt = parser.parse_args()

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('bi-conservative');
defs.epochs = opt.epochs


print(opt.evaluate)
print(opt.semsim)
# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']


def main():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('bi-conservative');
    defs.epochs = opt.epochs

    loss_fn, trainloader, validloader, _ = preprocess(opt, defs, valid=False)
    model = create_model(opt)

    if opt.semsim:
        model.fc = nn.Identity()

    # init model
    model.to(**setup)
    save_dir = create_checkpoint_dir(opt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if opt.semsim:
        file = f'{save_dir}/{arch}_{defs.epochs}_re_id.pth'
        inversefed.train_pl(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir, opt=opt)
        torch.save(model.state_dict(), f'{file}')
        model.eval()
        evaluate_semsim()
    else:
        file = f'{save_dir}/{arch}_{defs.epochs}.pth'
        inversefed.train_pl(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir, opt=opt)
        torch.save(model.state_dict(), f'{file}')
        model.eval()
        evaluate_class()
        evaluate()
        # inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir)


def evaluate():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('bi-conservative');
    defs.epochs = opt.epochs
    if opt.arch not in ['vit']:
        loss_fn, trainloader, validloader, _ = preprocess(opt, defs, valid=False)
        model = create_model(opt)
    else:
        loss_fn, trainloader, validloader, model = vit_preprocess(opt, defs, valid=False)
    model.to(**setup)
    root = create_checkpoint_dir(opt)

    filename = os.path.join(root, '{}_{}.pth'.format(opt.arch, opt.epochs))
    print(filename)
    if not os.path.exists(filename):
        assert False

    print(filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    stats = {'valid_losses': list(), 'valid_Accuracy': list(), 'valid_label': list(), 'valid_score': list()}

    if opt.arch == 'LeNet':
         inversefed.training.training_routine.validate_binary(model, loss_fn, validloader, defs, setup=setup, stats=stats)
    else:
         inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup=setup, stats=stats)

    print('acc_{}, loss_{}'.format(stats['valid_Accuracy'], stats['valid_losses']))


def evaluate_class():
        setup = inversefed.utils.system_startup()
        defs = inversefed.training_strategy('bi-conservative');
        defs.epochs = opt.epochs

        loss_fn, trainloader, validloader, reidloader = preprocess(opt, defs, valid=False)
        if opt.targte_data == 'CIFAR100':
            opt.rec_data_dir = 'benchmark/images/Cifar_ori_600/data_used_generate_RI'
        elif opt.targte_data == 'Caltech101':
            opt.rec_data_dir = 'benchmark/images/Caltech101/data_Caltech101_test_image'
        elif opt.targte_data == 'CelebA_Identity':
            opt.rec_data_dir = 'benchmark/images/CelebA_Identity/ResNet20-4_100_inversed_cele_aug_auglist__rlabel_False_ori'
        elif opt.targte_data == 'Dog':
            opt.rec_data_dir = 'benchmark/images/Dog/ResNet20-4_100_inversed_Dog_aug_auglist__rlabel_False_ori'
        elif opt.targte_data == 'ImageNette':
            opt.rec_data_dir = 'benchmark/images/ImageNette/data_ImageNette_test_image'

        _, _, _, reidloader_ori = preprocess(opt, defs, valid=False)
        model = create_model(opt)

        model.to(**setup)
        root = create_checkpoint_dir(opt)

        filename = os.path.join(root, '{}_{}.pth'.format(opt.arch, opt.epochs))
        print(filename)
        if not os.path.exists(filename):
            assert False

        print(filename)
        model.load_state_dict(torch.load(filename))
        model.fc = nn.Identity()
        model.eval()

        feature, filenames = inversefed.training.training_routine.validate_reid(model, loss_fn, reidloader, defs,
                                                                                setup=setup)
        feature_ori, filenames_ori = inversefed.training.training_routine.validate_reid(model, loss_fn, reidloader_ori,
                                                                                        defs, setup=setup)
        filenames_ori = sorted(filenames_ori, key=lambda p: int(str(p).split('_')[0]))
        feature_rec = []
        for i, name in enumerate(filenames_ori):
            rec_name = name.replace("_ori.png", "_rec.png")
            if rec_name in filenames:
                idx = filenames.index(rec_name)
                feature_rec.append(feature[idx])
        if feature_rec == []:
            for i, name in enumerate(filenames_ori):
                rec_name = name.replace("_ori.png", "_ori.png")
                if rec_name in filenames:
                    idx = filenames.index(rec_name)
                    feature_rec.append(feature[idx])

        feature_raw = np.stack(feature_ori)
        feature_rec = np.stack(feature_rec)
        distance_pos = np.mean(np.linalg.norm(feature_raw - feature_rec, axis=1))

        print(f"class_distance {distance_pos}")
        return np.linalg.norm(feature_raw - feature_rec, axis=1)

def tsevaluate_semsim():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('bi-conservative');
    defs.epochs = opt.epochs

    # reconstructed data
    loss_fn, trainloader, validloader, reidloader = preprocess(opt, defs, valid=False)

    # orginal data
    if opt.targte_data == 'CIFAR100':
        opt.rec_data_dir = 'benchmark/images/Cifar_ori_600/data_used_generate_RI'
    elif opt.targte_data == 'Caltech101':
        opt.rec_data_dir = 'benchmark/images/Caltech101/data_Caltech101_test_image'
    elif opt.targte_data == 'CelebA_Identity':
        opt.rec_data_dir = 'benchmark/images/CelebA_Identity/ResNet20-4_100_inversed_cele_aug_auglist__rlabel_False_ori'
    elif opt.targte_data == 'Dog':
        opt.rec_data_dir = 'benchmark/images/Dog/ResNet20-4_100_inversed_Dog_aug_auglist__rlabel_False_ori'
    elif opt.targte_data == 'ImageNette':
        opt.rec_data_dir = 'benchmark/images/ImageNette/data_ImageNette_test_image'
    _, _, _, reidloader_ori = preprocess(opt, defs, valid=False)


    model = create_model(opt)

    model.fc = nn.Identity()

    model.to(**setup)
    root = create_checkpoint_dir(opt)

    filename = os.path.join(root, '{}_{}_re_id.pth'.format(opt.arch, opt.epochs))
    print(filename)
    if not os.path.exists(filename):
        assert False

    print(filename)
    model.load_state_dict(torch.load(filename))
    model.eval()


    feature, filenames = inversefed.training.training_routine.validate_reid(model, loss_fn, reidloader, defs, setup=setup)
    feature_ori, filenames_ori = inversefed.training.training_routine.validate_reid(model, loss_fn, reidloader_ori, defs, setup=setup)
    print('feature_extracted')

    filenames_ori = sorted(filenames_ori, key=lambda p: int(str(p).split('_')[0]))

    feature_rec=[]
    for i,  name in enumerate(filenames_ori):
        rec_name = name.replace("_ori.png", "_rec.png")
        if rec_name in filenames:
            idx = filenames.index(rec_name)
            feature_rec.append(feature[idx])
    if feature_rec ==[]:
        for i, name in enumerate(filenames_ori):
            rec_name = name.replace("_ori.png", "_ori.png")
            if rec_name in filenames:
                idx = filenames.index(rec_name)
                feature_rec.append(feature[idx])

    feature_raw = np.stack(feature_ori)
    feature_rec = np.stack(feature_rec)

    distance_pos = np.mean(np.linalg.norm(feature_raw - feature_rec, axis=1))

    print(f"semsim_distance {distance_pos}")
    return np.linalg.norm(feature_raw - feature_rec, axis=1)

if __name__ == '__main__':

    if opt.targte_data == 'cifar100':
        root_dir = 'benchmark/images/cifar100/'
        with open('metrics/folder_names_cifar.txt', 'r') as f:
            lines = f.read().splitlines()
    elif opt.targte_data == 'Caltech101':
        root_dir = 'benchmark/images/Caltech101/'
        with open('metrics/folder_names_cal.txt', 'r') as f:
            lines = f.read().splitlines()
    elif opt.targte_data == 'CelebA_Identity':
        root_dir = 'benchmark/images/CelebA_Identity/'
        with open('metrics/folder_names_celeba.txt', 'r') as f:
            lines = f.read().splitlines()
    elif opt.targte_data == 'ImageNette':
        root_dir = 'benchmark/images/ImageNette/'
        with open('metrics/folder_names_imagenette.txt', 'r') as f:
            lines = f.read().splitlines()
    elif opt.targte_data == 'Dog':
        root_dir = 'benchmark/images/Dog/'
        with open('metrics/folder_names_dog.txt', 'r') as f:
            lines = f.read().splitlines()

    if opt.evaluate and opt.semsim:
        for i in range(len(lines)):
            opt.rec_data_dir = root_dir + lines[i]
            feature_dis = evaluate_semsim()
            if opt.semsim_psnr == True:
                  np.save(os.path.join(root_dir + lines[i], "semsim_psnr_feature.npy"), feature_dis)
            else:
                  np.save(os.path.join(root_dir + lines[i], "semsim_feature.npy"), feature_dis)
        exit(0)
    if opt.evaluate and opt.semsim == False:
        for i in range(len(lines)):
            opt.rec_data_dir = root_dir + lines[i]
            feature_dis = evaluate_class()
            np.save(os.path.join(root_dir + lines[i], "class_feature.npy"), feature_dis)
        exit(0)
    main()
