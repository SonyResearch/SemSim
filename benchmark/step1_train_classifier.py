import os, sys
sys.path.insert(0, './')
import torch

seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import inversefed
import argparse

import policy

from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess, create_checkpoint_dir

policies = policy.policies

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')

# not use in attack
parser.add_argument('--rec_data_dir', default='assets',
                    required=False, type=str, help='dir_of_rec_data to be test the acc')
parser.add_argument('--semsim', default=False, type=bool, help='semsim')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--aug_list', default='', required=False, type=str, help='Augmentation method.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='remove label.')
parser.add_argument('--evaluate', default=False, type=bool, help='Evaluate')
parser.add_argument('--pretrained', default=False, type=bool, help='Pretrained')

parser.add_argument('--defense', default=None, type=str, help='Existing Defenses')
parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')

opt = parser.parse_args()

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative');
defs.epochs = opt.epochs

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']


def main():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative');
    defs.epochs = opt.epochs
    loss_fn, trainloader, validloader, _ = preprocess(opt, defs, valid=False)
    model = create_model(opt)


    # init model
    model.to(**setup)
    save_dir = create_checkpoint_dir(opt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = f'{save_dir}/{arch}_{defs.epochs}.pth'

    inversefed.train_pl(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir, opt=opt)
    torch.save(model.state_dict(), f'{file}')
    model.eval()
    evaluate()


def evaluate():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative');
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

if __name__ == '__main__':
    if opt.evaluate:
        evaluate()
        exit(0)
    main()
