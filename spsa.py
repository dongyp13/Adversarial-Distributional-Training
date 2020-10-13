from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import datetime
import random

from models.wideresnet import *
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, help='model path')
# dataset dependent
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--samples_per_draw', default=256, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--sigma', default=1e-3, type=float)
parser.add_argument('--iterations', default=100, type=int)
parser.add_argument('--epsilon', default=8.0/255.0, type=float)
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--num_test', default=10000, type=int)

args = parser.parse_args()

if args.dataset == 'cifar10':
    print('------------cifar10---------')
    args.num_classes = 10
    args.image_size = 32
elif args.dataset == 'cifar100':
    print('----------cifar100---------')
    args.num_classes = 100
    args.image_size = 32
elif args.dataset == 'svhn':
    print('------------svhn10---------')
    args.num_classes = 10
    args.image_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([transforms.ToTensor(),])

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='../data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='../data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

elif args.dataset == 'svhn':
    testset = torchvision.datasets.SVHN(root='../data',
                                        split='test',
                                        download=True,
                                        transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=1)


basic_net = WideResNet(depth=28,
                       num_classes=args.num_classes,
                       widen_factor=10)

net = basic_net.to(device)
net.load_state_dict(torch.load(args.model_path))

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

        loss = -(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            loss = torch.mean(loss)

        return loss

def spsa_blackbox(model,
                  xs,
                  ys,
                  batch_size=args.batch_size,
                  samples_per_draw=args.samples_per_draw,
                  lr=args.learning_rate,
                  sigma=args.sigma,
                  iterations=args.iterations,
                  epsilon=args.epsilon,
                  num_classes=10,
                  logging=True):
    
      assert(len(xs.shape) == 3)
      assert((samples_per_draw // 2) % batch_size == 0)
      model.eval()
      pred = model(xs.unsqueeze(0)).max(1)[1]
      if pred != ys:
        return xs, False, 0

      xs_adv = Variable(xs.data.clone(), requires_grad=True).cuda()
      ys_repeat = ys.repeat(batch_size * 2)
      opt = torch.optim.Adam([xs_adv], lr=lr)
      for _ in range(1, iterations + 1):
          grad = 0
          for i in range(0, samples_per_draw // 2, batch_size):
              pert = (torch.rand([batch_size] + list(xs_adv.shape)) * 2 - 1).sign().cuda()
              pert = torch.cat([pert, -pert], 0)

              eval_points = xs_adv + sigma * pert
              with torch.no_grad():
                  losses = CWLoss(num_classes, reduce=False)(model(eval_points), ys_repeat)

              grad -= (losses.view([batch_size * 2, 1, 1, 1]) * pert).mean(0)

          grad = grad / sigma / ((samples_per_draw // 2) / batch_size)
          opt.zero_grad()
          xs_adv.backward(grad)
          opt.step()
          
          xs_adv.data = torch.min(torch.max(xs_adv.data, xs.data - epsilon), xs.data + epsilon)
          xs_adv.data.clamp_(0.0, 1.0)

          with torch.no_grad():
            xs_adv_label = model(xs_adv.unsqueeze(0)).max(1)[1]

          if logging:
              print("iteration:{}, loss:{}, learning rate:{}, "
                    "prediction:{}, distortion:{}".format(
                  _, losses.mean(), lr, xs_adv_label.item(),
                  torch.max(torch.abs(xs_adv - xs))
              ))

          if xs_adv_label != ys:
            return xs_adv, False, _

      return xs_adv, True, iterations + 1


per_image_acc = np.zeros([args.num_test])
queries = np.zeros([args.num_test])

for idx, (data, target) in enumerate(testloader):
    print('test sample: ', idx)
    data, target = data.to(device), target.to(device)
    X, y = Variable(data.squeeze(0), requires_grad=True), Variable(target)
    X_spsa, per_image_acc[idx], queries[idx] = spsa_blackbox(net, X, y)
    if idx == args.num_test - 1:
        break
print(np.sum(per_image_acc), np.mean(queries))

