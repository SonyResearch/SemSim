"""Implement the .train function."""

import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict

from .scheduler import GradualWarmupScheduler

from .. import consts
from ..consts import BENCHMARK, NON_BLOCKING
torch.backends.cudnn.benchmark = BENCHMARK
from tqdm import tqdm

def train(model, loss_fn, trainloader, validloader, defs, setup=dict(dtype=torch.float, device=torch.device('cpu')), save_dir=None):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, defs)
    print('starting to training model')
    for epoch in tqdm(range(defs.epochs)):
        model.train()
        step(model, loss_fn, trainloader, optimizer, scheduler, defs, setup, stats)

        if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
            model.eval()
            validate(model, loss_fn, validloader, defs, setup, stats)
            # Print information about loss and accuracy
            print_status(epoch, loss_fn, optimizer, stats)
            if save_dir is not None:
                file = f'{save_dir}/{epoch}.pth'
                torch.save(model.state_dict(), f'{file}')

        if defs.dryrun:
            break
        if not (np.isfinite(stats['train_losses'][-1])):
            print('Loss is NaN/Inf ... terminating early ...')
            break

    return stats

def step(model, loss_fn, dataloader, optimizer, scheduler, defs, setup, stats):
    """Step through one epoch."""
    dm = torch.as_tensor(consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(consts.cifar10_std, **setup)[:, None, None]

    epoch_loss, epoch_metric = 0, 0
    for batch, data_info in enumerate(dataloader):
        # if batch % 10 == 0: 
        #     print('run [{}/{}]'.format(batch, len(dataloader)))
        # Prep Mini-Batch
        optimizer.zero_grad()
        # Transfer to GPU
        if isinstance(data_info, dict):
            print('-------------------sxx------------------')
            for key in data_info.keys(): 
                data_info[key] = data_info[key].to('cuda')
            model_output = model(**data_info)
            # import pdb; pdb.set_trace() 
            loss = model_output.loss
            targets = data_info['labels']
            outputs = model_output.logits 
        else: 
            inputs, targets = data_info
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
            # Get loss
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)


        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if defs.dryrun:
            break
    if defs.scheduler == 'linear':
        scheduler.step()

    stats['train_losses'].append(epoch_loss / (batch + 1))
    stats['train_' + name].append(epoch_metric / (batch + 1))


def validate(model, loss_fn, dataloader, defs, setup, stats):
    """Validate model effectiveness of val dataset."""
    epoch_loss, epoch_metric = 0, 0
    with torch.no_grad():
        for batch, data_info in enumerate(dataloader):
            # Transfer to GPU
            if isinstance(data_info, dict):
                for key in data_info.keys(): 
                    data_info[key] = data_info[key].to('cuda')
                model_output = model(**data_info)
                # import pdb; pdb.set_trace() 
                loss = model_output.loss
                targets = data_info['labels']
                outputs = model_output.logits 
            else: 
                inputs, targets = data_info
                inputs = inputs.to(**setup)
                targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
                # Get loss
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                metric, name, _ = loss_fn.metric(outputs, targets)

                if len(outputs.t()) > 9:
                        top_k_index, softmax_score = get_top_k_index(outputs, 20)
                        stats['valid_label'].extend(top_k_index)
                        stats['valid_score'].extend(softmax_score.cpu().tolist())
                else:
                        top_k_index, softmax_score = get_top_k_index(outputs, len(outputs.t()))
                        stats['valid_label'].extend(top_k_index)
                        stats['valid_score'].extend(softmax_score.cpu().tolist())

                epoch_loss += loss.item()
                epoch_metric += metric.item()*targets.shape[0]

                if defs.dryrun:
                    break

    # stats['valid_losses'].append(epoch_loss / (batch + 1))
    # stats['valid_' + name].append(epoch_metric / (batch + 1))

    stats['valid_losses'].append(epoch_loss / (batch + 1))
    stats['valid_' + name].append(epoch_metric / len(dataloader.dataset))


def validate_binary(model, loss_fn, dataloader, defs, setup, stats):
    """Validate model effectiveness of val dataset."""
    epoch_loss, number_positive = 0, 0
    with torch.no_grad():
        for batch, data_info in enumerate(dataloader):
            # Transfer to GPU
            if isinstance(data_info, dict):
                for key in data_info.keys():
                    data_info[key] = data_info[key].to('cuda')
                model_output = model(**data_info)
                # import pdb; pdb.set_trace()
                loss = model_output.loss
                targets = data_info['labels']
                outputs = model_output.logits
            else:
                inputs, targets = data_info
                inputs = inputs.to(**setup)
                targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
                # Get loss
                outputs = model(inputs)
                loss = loss_fn(outputs, targets.float())
                epoch_loss += loss.item()
                number_positive += ((outputs > 0.5) == targets.float()).sum().item()

    stats['valid_losses'].append(epoch_loss / len(dataloader.dataset))
    stats['valid_Accuracy'].append(number_positive / len(dataloader.dataset))



def validate_reid(model, loss_fn, dataloader, defs, setup):
    """Validate model effectiveness of val dataset."""
    epoch_loss, number_positive = 0, 0
    feaure1=[]
    with torch.no_grad():
        for batch, data_info in enumerate(dataloader):
            # batch_filenames.append([dataloader.dataset.get_filename(i) for i in range(len(batch))])
            # filenames = dataloader.dataset.get_filename()[
            #             batch * dataloader.batch_size:(batch + 1) * dataloader.batch_size]
            # batch_filenames.append(filenames)
            # Transfer to GPU
            if isinstance(data_info, dict):
                for key in data_info.keys():
                    data_info[key] = data_info[key].to('cuda')
                model_output = model(**data_info)
                # import pdb; pdb.set_trace()
                loss = model_output.loss
                targets = data_info['labels']
                outputs = model_output.logits
            else:
                inputs, targets = data_info
                inputs = inputs.to(**setup)

                # targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
                # Get loss
                # distance_pos = torch.sqrt(torch.sum(torch.pow(anchor - inputs, 2), dim=1))
                outputs = model(inputs)
                outputs = outputs.cpu()
                feaure1.extend(outputs.numpy())
        filenames = dataloader.dataset.get_filename()

    return feaure1, filenames
    # stats['valid_losses'].append(epoch_loss / len(dataloader.dataset))
# stats['valid_Accuracy'].append(number_positive / len(dataloader.dataset))

def get_top_k_index(outputs, k):

    # confidence score and top-k labels
    score = nn.Softmax(dim=1)
    softmax_score = score(outputs)
    softmax_score.argmax()
    top_k_index = torch.topk(softmax_score, k).indices.cpu().tolist()

    return  top_k_index, softmax_score

def set_optimizer(model, defs):
    """Build model optimizer and scheduler from defs.

    The linear scheduler drops the learning rate in intervals.
    # Example: epochs=160 leads to drops at 60, 100, 140.
    """
    print(f'Learning rate {defs.lr}')
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
        # Scheduler is fixed to 120 epochs so that calls with fewer epochs are equal in lr drops.

    if defs.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)

    return optimizer, scheduler


def print_status(epoch, loss_fn, optimizer, stats):
    """Print basic console printout every defs.validation epochs."""
    current_lr = optimizer.param_groups[0]['lr']
    name, format = loss_fn.metric()
    print(f'Epoch: {epoch}| lr: {current_lr:.4f} | '
          f'Train loss is {stats["train_losses"][-1]:6.4f}, Train {name}: {stats["train_" + name][-1]:{format}} | '
          f'Val loss is {stats["valid_losses"][-1]:6.4f}, Val {name}: {stats["valid_" + name][-1]:{format}} |')


def prune(gradient, percent):
    k = int(gradient.numel() * percent * 0.01)
    shape = gradient.shape
    gradient = gradient.flatten()
    index = torch.topk(torch.abs(gradient), k, largest=False)[1]
    gradient[index] = 0.
    gradient = gradient.view(shape)
    return gradient


def add_noise(model, lr):
    for param in model.parameters():
        param.grad.data += lr * torch.randn(param.grad.data.shape).cuda()


def lap_sample(shape):
    from torch.distributions.laplace import Laplace
    m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    return m.expand(shape).sample()

def lap_noise(model, lr):
    for param in model.parameters():
        param.grad.data += lr * lap_sample(param.grad.data.shape).cuda()


def global_prune(model, percent):
    for param in model.parameters():
        param.grad.data = prune(param.grad.data, percent)


def step_with_defense(model, loss_fn, dataloader, optimizer, scheduler, defs, setup, stats, opt):
    """Step through one epoch."""
    assert opt is not None
    dm = torch.as_tensor(consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(consts.cifar10_std, **setup)[:, None, None]

    epoch_loss, epoch_metric = 0, 0
    for batch, (inputs, targets) in enumerate(dataloader):
        # Prep Mini-Batch
        optimizer.zero_grad()
        # Transfer to GPU
        inputs = inputs.to(**setup)
        targets = targets.to(device=setup['device'], non_blocking=NON_BLOCKING)
        # Get loss
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        epoch_loss += loss.item()
        loss.backward()


        if 'gaussian' in opt.defense:
            if '1e-3' in opt.defense:
                add_noise(model, 1e-3)
            elif '1e-2' in opt.defense:
                add_noise(model, 1e-2)
            else:
                raise NotImplementedError
        elif 'lap' in opt.defense:
            if '1e-3'  in opt.defense:
                lap_noise(model, 1e-3)
            elif '1e-2' in opt.defense:
                lap_noise(model, 1e-2)
            elif '1e-1' in opt.defense:
                lap_noise(model, 1e-1)
            else:
                raise NotImplementedError
        
        elif 'prune' in opt.defense:
            found = False
            for i in [10, 20, 30, 50, 70, 80, 90, 95, 99]:
                if str(i) in opt.defense:
                    found=True
                    global_prune(model, i)

            if not found:
                raise NotImplementedError


        optimizer.step()

        metric, name, _ = loss_fn.metric(outputs, targets)
        epoch_metric += metric.item()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if defs.dryrun:
            break
    if defs.scheduler == 'linear':
        scheduler.step()

    stats['train_losses'].append(epoch_loss / (batch + 1))
    stats['train_' + name].append(epoch_metric / (batch + 1))




def train_with_defense(model, loss_fn, trainloader, validloader, defs, setup=dict(dtype=torch.float, device=torch.device('cpu')), save_dir=None, opt=None):
    """Run the main interface. Train a network with specifications from the Strategy object."""
    assert opt is not None
    stats = defaultdict(list)
    optimizer, scheduler = set_optimizer(model, defs)
    print('starting to training model')
    for epoch in range(defs.epochs):
        model.train()
        step_with_defense(model, loss_fn, trainloader, optimizer, scheduler, defs, setup, stats, opt)

        if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
            model.eval()
            validate(model, loss_fn, validloader, defs, setup, stats)
            # Print information about loss and accuracy
            print_status(epoch, loss_fn, optimizer, stats)
            if save_dir is not None:
                file = f'{save_dir}/{epoch}.pth'
                torch.save(model.state_dict(), f'{file}')

        if defs.dryrun:
            break
        if not (np.isfinite(stats['train_losses'][-1])):
            print('Loss is NaN/Inf ... terminating early ...')
            break

    return stats
