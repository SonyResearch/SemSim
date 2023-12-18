"""Repeatable code parts concerning data loading."""


import torch
import torchvision
import torchvision.transforms as transforms

import os

from ..consts import *

from .data import _build_bsds_sr, _build_bsds_dn
from .loss import Classification, PSNR, TripletLoss, TripletLoss_PSNR
from .datasets import CelebAForGender, CelebAForMLabel, CelebAForSmile, CelebAFaceAlignForMLabel, DatasetFromFolder
from inversefed.data.rec_dataset import RecDataset

from inversefed.data.re_id_dataset import RE_ID
from inversefed.data.re_id_dataset_psnr import RE_ID_PSNR
from inversefed.data.re_id_dataset_cross_tri import RE_ID_Tri_All_Data
def construct_dataloaders(dataset, defs, rec_data_dir, data_path='~/data', shuffle=True, normalize=True, opt = []):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    if opt == []:
        arch = []
    else:
        arch = opt.arch
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR10':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CIFAR100':
        trainset, validset, recset = _build_cifar100(path, rec_data_dir, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST':
        trainset, validset = _build_mnist(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST_GRAY':
        trainset, validset = _build_mnist_gray(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'human_anno':
        trainset, validset, recset = _build_human_anno(path, rec_data_dir, defs.augmentations, normalize)
        if arch == 'LeNet':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = Classification()
    elif dataset == 'human_anno_id':
        trainset, validset, recset = _build_human_anno_id(path, rec_data_dir, defs.augmentations, normalize, opt)
        if arch == 'LeNet':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = Classification()
        if opt.semsim:
           if opt.semsim_psnr == True:
               loss_fn = TripletLoss_PSNR()
           else:
                loss_fn = TripletLoss()
    elif dataset == 'Caltech101':
        trainset, validset, recset = _build_caltech101(path, rec_data_dir, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'Dog':
        trainset, validset, recset = _build_dog(path, rec_data_dir, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'ImageNette':
        trainset, validset, recset = _build_imagenette(path, rec_data_dir, defs.augmentations, normalize)
        loss_fn = Classification()

    elif dataset == 'ImageNet':
        trainset, validset = _build_imagenet(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CelebA_Gender':
        trainset, validset = _build_celeba_gender(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CelebA_Smile':
        trainset, validset = _build_celeba_smile(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CelebA_Identity':
        trainset, validset, recset = _build_celeba_identity(path, rec_data_dir, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CelebA_MLabel':
        trainset, validset = _build_celeba_mlabel(path, defs.augmentations, normalize)
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif dataset == 'CelebAFaceAlign_MLabel':
        trainset, validset = _build_celeba_face_align_mlabel(path, defs.augmentations, normalize)
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif dataset == 'BSDS-SR':
        trainset, validset = _build_bsds_sr(path, defs.augmentations, normalize, upscale_factor=3, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'BSDS-DN':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=False)
        loss_fn = PSNR()
    elif dataset == 'BSDS-RGB':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=True)
        loss_fn = PSNR()

    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    recloader = torch.utils.data.DataLoader(recset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return loss_fn, trainloader, validloader, recloader


def _build_cifar10(data_path, augmentations=True, normalize=True):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_cifar100(data_path, rec_data_dir, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    #
    recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    recset.transform = transform
    return trainset, validset, recset


def _build_human_anno(data_path, rec_data_dir, augmentations=True, normalize=True):
    """Define human_anno with everything considered."""
    # Load data
    data_path = '../../data/human_anno'
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageFolder(root=data_path + '/val', transform=transforms.ToTensor())
    #
    recset = torchvision.datasets.ImageFolder(root=data_path + '/' + rec_data_dir, transform=transforms.ToTensor())
    # recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    recset.transform = transform
    return trainset, validset, recset


def _build_human_anno_id(data_path, rec_data_dir, augmentations=True, normalize=True, opt = []):
    """Define CIFAR-100 with everything considered."""
    # Load data
    data_path = '../../../data/human_anno_id'
    if opt.semsim == False:
        trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transforms.ToTensor())
        validset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transforms.ToTensor())
        recset =  RecDataset(rec_data_dir, transform=transforms.ToTensor())
    else:
        if opt.semsim_psnr == False:
            # trainset = RE_ID(data_path, transform=transforms.ToTensor())
            # validset = RE_ID(data_path, transform=transforms.ToTensor())
            trainset = RE_ID_Tri_All_Data(data_path +  '/train_with_ori', transform=transforms.ToTensor())
            validset = RE_ID_Tri_All_Data(data_path +  '/train_with_ori', transform=transforms.ToTensor())
            recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())
        else:
            trainset = RE_ID_PSNR(data_path, transform=transforms.ToTensor())
            validset = RE_ID_PSNR(data_path, transform=transforms.ToTensor())
            recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())



    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    recset.transform = transform
    return trainset, validset, recset

def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_caltech101(data_path, rec_data_dir, augmentations=True, normalize=True):
    """Define Caltech101 with everything considered."""
    # Load data
    data_path = '../../data/caltech-101'
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageFolder(root=data_path + '/test', transform=transforms.ToTensor())
    #
    recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())
    if caltech101_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = caltech101_mean, caltech101_std

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    recset.transform = transform
    return trainset, validset, recset


def _build_imagenette(data_path, rec_data_dir, augmentations=True, normalize=True):
    # Load data
    data_path = '../../data/imagenette2'
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageFolder(root=data_path + '/val', transform=transforms.ToTensor())
    #
    recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())
    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomCrop(112, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    recset.transform = transform
    return trainset, validset, recset


def _build_dog(data_path, rec_data_dir, augmentations=True, normalize=True):
    # Load data
    data_path = '../../data/stanford_dog'
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageFolder(root=data_path + '/test', transform=transforms.ToTensor())
    #
    recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())
    if caltech101_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    recset.transform = transform
    return trainset, validset, recset


def _build_imagenet(data_path, augmentations=True, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    # trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    # validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())
    data_path = os.path.join(data_path, 'imagenet-split-0') if not os.path.exists('/home/zx/nfs/server3/data/imagenet-split-0') else '/home/zx/nfs/server3/data/imagenet-split-0'
    # data_path = '/home/zx/data/imagenet-split-0'
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageFolder(root=data_path + '/val', transform=transforms.ToTensor())

    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_celeba_gender(data_path, augmentations=True, normalize=True):
    """Define celeba with everything considered."""
    # Load data
    # trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    # validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    data_path = data_path if not os.path.exists('/home/zx/nfs/server3/data/') else '/home/zx/nfs/server3/data/'
    # data_path = '/home/zx/data'
    trainset = CelebAForGender(data_path, split='train', transform=transforms.ToTensor())
    validset = CelebAForGender(data_path, split='valid', transform=transforms.ToTensor())

    if celeba_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = celeba_mean, celeba_std

    # Organize preprocessing
    transform = transforms.Compose([
        # transforms.Resize((128,128)),
        transforms.Resize((112,112)),
        # transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_celeba_smile(data_path, augmentations=True, normalize=True):
    """Define celeba with everything considered."""
    # Load data
    # trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    # validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    data_path = data_path if not os.path.exists('/home/zx/nfs/server3/data/') else '/home/zx/nfs/server3/data/'
    data_path = '/home/zx/data'
    trainset = CelebAForSmile(data_path, split='train', transform=transforms.ToTensor())
    validset = CelebAForSmile(data_path, split='valid', transform=transforms.ToTensor())

    if celeba_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = celeba_mean, celeba_std

    # Organize preprocessing
    transform = transforms.Compose([
        # transforms.Resize((128,128)),
        transforms.Resize((112,112)),
        # transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_celeba_identity(data_path,rec_data_dir, augmentations=True, normalize=True):
    """Define celeba with everything considered."""
    # Load data
    # data_path = data_path if not os.path.exists('/home/zx/nfs/server3/data/celeba') else '/home/zx/nfs/server3/data/celeba'
    data_path = '../../data/celeba'

    # data_path = '/home/zx/data/celeba'
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'id_500_split/train'),  transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'id_500_split/test'), transform=transforms.ToTensor())
    recset = RecDataset(rec_data_dir, transform=transforms.ToTensor())

    if celeba_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = celeba_mean, celeba_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        # transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    recset.transform = transform
    return trainset, validset, recset

def _build_celeba_mlabel(data_path, augmentations=True, normalize=True):
    """Define celeba with everything considered."""
    # Load data
    data_path = data_path if not os.path.exists('/home/zx/nfs/server3/data/celeba') else '/home/zx/nfs/server3/data/celeba'

    # data_path = '/home/zx/data/'
    trainset = CelebAForMLabel(data_path, split='train', transform=transforms.ToTensor())
    validset = CelebAForMLabel(data_path, split='valid', transform=transforms.ToTensor())

    if celeba_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = celeba_mean, celeba_std

    # Organize preprocessing
    transform = transforms.Compose([
        # transforms.Resize((128,128)),
        transforms.Resize((112,112)),
        # transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_celeba_face_align_mlabel(data_path, augmentations=True, normalize=True):
    """Define celeba with everything considered."""
    # Load data
    data_path = data_path if not os.path.exists('/home/zx/nfs/server3/data/celeba') else '/home/zx/nfs/server3/data/celeba'

    # data_path = '/home/zx/data/'
    trainset = CelebAFaceAlignForMLabel(data_path, split='train', transform=transforms.ToTensor())
    validset = CelebAFaceAlignForMLabel(data_path, split='valid', transform=transforms.ToTensor())

    if celeba_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = celeba_mean, celeba_std

    # Organize preprocessing
    transform = transforms.Compose([
        # transforms.Resize((128,128)),
        transforms.Resize((112,112)),
        # transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _get_meanstd(dataset):
    cc = torch.cat([dataset[i][0].reshape(3, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std
