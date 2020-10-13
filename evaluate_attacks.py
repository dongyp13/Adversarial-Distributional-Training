from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=8.0/255.0,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=2.0/255.0,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--attack-method', default='PGD')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
args = parser.parse_args()
print(args)

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'svhn':
    args.epsilon = 4.0 / 255.0
    args.step_size = 1.0 / 255.0
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
else:
    raise NotImplementedError

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _cw_whitebox(model,
                 X,
                 y,
                 epsilon=args.epsilon,
                 num_steps=args.num_steps,
                 step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = CWLoss(100 if args.dataset == 'cifar100' else 10)(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err cw (white-box): ', err_pgd)
    return err, err_pgd


def _fgsm_whitebox(model,
                   X,
                   y,
                   epsilon=args.epsilon):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_fgsm = Variable(X.data, requires_grad=True)

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()

    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)
    err_pgd = (model(X_fgsm).data.max(1)[1] != y.data).float().sum()
    print('err fgsm (white-box): ', err_pgd)
    return err, err_pgd


def _mim_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  decay_factor=1.0):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    
    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err mim (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def _cw_blackbox(model_target,
                 model_source,
                 X,
                 y,
                 epsilon=args.epsilon,
                 num_steps=args.num_steps,
                 step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = CWLoss(100 if args.dataset == 'cifar100' else 10)(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err cw (black-box): ', err_pgd)
    return err, err_pgd

def _fgsm_blackbox(model_target,
                   model_source,
                   X,
                   y,
                   epsilon=args.epsilon):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_fgsm = Variable(X.data, requires_grad=True)

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model_source(X_fgsm), y)
    loss.backward()

    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_fgsm).data.max(1)[1] != y.data).float().sum()
    print('err fgsm (black-box): ', err_pgd)
    return err, err_pgd

def _mim_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  decay_factor=1.0):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    
    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err mim (white-box): ', err_pgd)
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader, attack_method):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        if attack_method == 'PGD':
            err_natural, err_robust = _pgd_whitebox(model, X, y)
        elif attack_method == 'CW':
            err_natural, err_robust = _cw_whitebox(model, X, y)
        elif attack_method == 'MIM':
            err_natural, err_robust = _mim_whitebox(model, X, y)
        elif attack_method == 'FGSM':
            err_natural, err_robust = _fgsm_whitebox(model, X, y)
        else:
            raise NotImplementedError
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def eval_adv_test_blackbox(model_target, model_source, device, test_loader, attack_method):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        if attack_method == 'PGD':
            err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        elif attack_method == 'CW':
            err_natural, err_robust = _cw_blackbox(model_target, model_source, X, y)
        elif attack_method == 'MIM':
            err_natural, err_robust = _mim_blackbox(model_target, model_source, X, y)
        elif attack_method == 'FGSM':
            err_natural, err_robust = _fgsm_blackbox(model_target, model_source, X, y)
        else:
            raise NotImplementedError
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():

    if args.white_box_attack:
        # white-box attack
        print('white-box attack')
        model = WideResNet(depth=28, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model.load_state_dict(torch.load(args.model_path))

        eval_adv_test_whitebox(model, device, test_loader, args.attack_method)
    else:
        # black-box attack
        print('black-box attack')
        model_target = WideResNet(depth=28, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet(depth=28, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader, args.attack_method)


if __name__ == '__main__':
    main()
