# This code is adopted from "https://github.com/Line290/FeatureAttack"

from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

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
                    help='dataset')  # concat cascade
parser.add_argument('--batch_size_test',
                    default=200,
                    type=int,
                    help='batch size for testing')
parser.add_argument('--image_size', default=32, type=int, help='image size')

args = parser.parse_args()

if args.dataset == 'cifar10':
    print('------------cifar10---------')
    args.num_classes = 10
    args.image_size = 32
    epsilon = 8.0/255.0
elif args.dataset == 'cifar100':
    print('----------cifar100---------')
    args.num_classes = 100
    args.image_size = 32
    epsilon = 8.0/255.0
elif args.dataset == 'svhn':
    print('------------svhn10---------')
    args.num_classes = 10
    args.image_size = 32
    epsilon = 8.0/255.0


device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print('==> Preparing data..')

if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
elif args.dataset == 'svhn':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

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
                                         batch_size=10000,
                                         shuffle=False,
                                         num_workers=20)


basic_net = WideResNet(depth=28,
                       num_classes=args.num_classes,
                       widen_factor=10)

net = basic_net.to(device)
net.load_state_dict(torch.load(args.model_path))

criterion = nn.CrossEntropyLoss()

config_feature_attack = {
    'train': False,
    'epsilon': epsilon,
    'num_steps': 50,
    'step_size': 1.0 / 255.0,
    'random_start': True,
    'early_stop': True,
    'num_total_target_images': args.batch_size_test,
}

def pair_cos_dist(x, y):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    c = torch.clamp(1 - cos(x, y), min=0)
    return c

def attack(model, inputs, target_inputs, y, config):
    step_size = config['step_size']
    epsilon = config['epsilon']
    num_steps = config['num_steps']
    random_start = config['random_start']
    early_stop = config['early_stop']
    model.eval()

    x = inputs.detach()
    if random_start:
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x = torch.clamp(x, 0.0, 1.0)

    target_logits, target_feat = model(target_inputs, return_feature=True)
    target_feat = target_feat.detach()

    for i in range(num_steps):
        x.requires_grad_()
        zero_gradients(x)
        if x.grad is not None:
            x.grad.data.fill_(0)
        logits_pred, feat = model(x, return_feature=True)
        preds = logits_pred.argmax(1)
        if early_stop:
            num_not_corr = (preds != y).sum().item()
            if num_not_corr > 0:
                break
        inver_loss = pair_cos_dist(feat, target_feat)
        adv_loss = inver_loss.mean()
        adv_loss.backward()
        x_adv = x.data - step_size * torch.sign(x.grad.data)
        x_adv = torch.min(torch.max(x_adv, inputs - epsilon), inputs + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x = Variable(x_adv)
    return x.detach(), preds


target_images_size = args.batch_size_test
print('target batch size is: ', target_images_size)
num_total_target_images = config_feature_attack['num_total_target_images']

net.eval()

untarget_success_count = 0
target_success_count = 0
total = 0
# load all test data
all_test_data, all_test_label = None, None
for test_data, test_label in testloader:
    all_test_data, all_test_label = test_data, test_label
print(all_test_data.size(), all_test_label.size())

num_eval_imgs = all_test_data.size(0)
per_image_acc = np.zeros([num_eval_imgs])
for clean_idx in range(num_eval_imgs):
    input, label_cpu = all_test_data[clean_idx].unsqueeze(0), all_test_label[clean_idx].unsqueeze(0)
    start_time = time.time()
    batch_idx_list = {}
    other_label_test_idx = (all_test_label != label_cpu[0])
    other_label_test_data = all_test_data[other_label_test_idx]
    other_label_test_label = all_test_label[other_label_test_idx]
    num_other_label_img = other_label_test_data.size(0)

    # Setting candidate targeted images
    candidate_indices = torch.zeros(num_total_target_images).long().random_(0, num_other_label_img)
    num_batches = int(math.ceil(num_total_target_images / target_images_size))
    # print(other_label_test_idx.size(), other_label_test_data.size(), other_label_test_label.size())

    # Init index of image which be attacked successfully
    adv_idx = 0

    for i in range(num_batches):
        bstart = i * target_images_size
        bend = min(bstart + target_images_size, num_total_target_images)

        target_inputs = other_label_test_data[candidate_indices[bstart:bend]]
        target_labels_cpu = other_label_test_label[candidate_indices[bstart:bend]]
        target_inputs, target_labels = target_inputs.to(device), target_labels_cpu.to(device)

        input, label = input.to(device), label_cpu.to(device)
        inputs = input.repeat(target_images_size, 1, 1, 1)
        labels = label.repeat(target_images_size)


        # print(inputs.size(), labels)
        # print(target_inputs.size(), target_labels)
        x_batch_adv, predicted = attack(net, inputs, target_inputs, labels, config_feature_attack)
        print((x_batch_adv - inputs).max(), (x_batch_adv - inputs).min())

        # print(predicted.size())
        not_correct_idices = (predicted != labels).nonzero().view(-1)
        not_corrent_num = not_correct_idices.size(0)
        attack_success_num = predicted.eq(target_labels).sum().item()

        per_image_acc[clean_idx] = (not_corrent_num == 0)
        # At least one misclassified
        if not_corrent_num != 0:
            untarget_success_count += 1
            if attack_success_num != 0:
                target_success_count += 1
            adv_idx = not_correct_idices[0]
            break

    total += 1
    duration = time.time() - start_time
    #x_adv.append(x_batch_adv[adv_idx].unsqueeze(0).cpu())

    
    print(
        "step %d, duration %.2f, aver untargeted attack success %.2f, aver targeted attack success %.2f"
        % (clean_idx, duration, 100. * untarget_success_count / total, 100.*target_success_count / total))
    sys.stdout.flush()

acc = 100. * untarget_success_count / total
print('Val acc:', acc)
print('Storing examples')
