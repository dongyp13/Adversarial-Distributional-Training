from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.distributions.utils import clamp_probs
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from models.generator import define_G, get_scheduler, set_requires_grad, Encoder

parser = argparse.ArgumentParser(description='PyTorch Adversarial Distributional Training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8.0/255.0,
                    help='perturbation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
# parameters for the generator
parser.add_argument('--net_G', type=str, default='resnet_3blocks', 
                    help='net for G')
parser.add_argument('--opt_G', type=str, default='adam', 
                    help='optimizer for G')
parser.add_argument('--lr_G', type=float, default=0.0002, 
                    help='initial learning rate for adam')
parser.add_argument('--lr_policy_G', type=str, default='linear', 
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters_G', type=int, default=30, 
                    help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--niter_G', type=int, default=100, 
                    help='# of iter at starting learning rate')
parser.add_argument('--niter_decay_G', type=int, default=50, 
                    help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1_G', type=float, default=0.5, 
                    help='momentum term of adam')
parser.add_argument('--ngf_G', type=int, default=256, 
                    help='# ')
parser.add_argument('--z_dim', type=int, default=64, 
                    help='z_dim')
parser.add_argument('--lbd', type=float, default=1., 
                    help='lambda for the entropy term')
parser.add_argument('--entropy_th', type=float, default=0.9, 
                    help='threshold for the entropy term')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    help='dataset')

args = parser.parse_args()
print(args)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
elif args.dataset == 'svhn':
    args.epsilon = 4.0 / 255.0
    trainset = torchvision.datasets.SVHN(root='../data', split='train', download=True, transform=transform_test)
    testset = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform_test)
else:
    raise NotImplementedError
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def grad_inv(grad):
    return grad.neg()

def train(args, model, device, train_loader, optimizer, epoch, G, optimizer_G, encoder, optimizer_encoder):
    model.train()
    G.train()
    encoder.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # calculate the two-step gradients as inputs to the generator
        model.eval()
        data.requires_grad_()
        loss_ = F.cross_entropy(model(data), target)
        grad = torch.autograd.grad(loss_, [data])[0].detach()
        data.requires_grad_(False)

        x_fgsm = torch.clamp(data + args.epsilon * grad.sign(), 0.0, 1.0).detach()
        x_fgsm.requires_grad_()
        grad_fgsm = torch.autograd.grad(F.cross_entropy(model(x_fgsm), target), [x_fgsm])[0].detach()
        x_fgsm.requires_grad_(False)

        rand_z = torch.rand(data.size(0), args.z_dim, device='cuda') * 2. - 1.
        pert = G(torch.cat([data, grad, grad_fgsm], 1), z=rand_z).tanh()

        model.train()
        optimizer.zero_grad()
        optimizer_G.zero_grad()
        optimizer_encoder.zero_grad()

        logits_z = encoder(pert)
        mean_z, var_z = logits_z[:, :args.z_dim], F.softplus(logits_z[:, args.z_dim:])
        neg_entropy_ub = -(-((rand_z - mean_z) ** 2) / (2 * var_z+1e-8) - (var_z+1e-8).log()/2. - math.log(math.sqrt(2 * math.pi))).mean(1).mean(0)

        x_adv = torch.clamp(data + args.epsilon * torch.clamp(pert, -1, 1), 0.0, 1.0)
        x_adv.register_hook(grad_inv)
        loss = F.cross_entropy(model(x_adv), target)

        (loss + args.lbd * F.relu(neg_entropy_ub - args.entropy_th)).backward()
        optimizer.step()
        optimizer_G.step()
        optimizer_encoder.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    model = WideResNet(depth=28, num_classes=100 if args.dataset == 'cifar100' else 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    G = define_G(10, 3, args.ngf_G, args.net_G, z_dim=args.z_dim)
    
    if args.opt_G == 'adam':
        optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=(args.beta1_G, 0.999))
    elif args.opt_G == 'sgd':
        optimizer_G = torch.optim.SGD(G.parameters(), lr=args.lr_G, weight_decay=1e-4)
    elif args.opt_G == 'momentum':
        optimizer_G = torch.optim.SGD(G.parameters(), lr=args.lr_G, momentum=0.9, weight_decay=1e-4)
    elif args.opt_G == 'rmsprop':
        optimizer_G = torch.optim.RMSprop(G.parameters(), lr=args.lr_G)
    else:
        raise NotImplementedError
    scheduler_G = get_scheduler(optimizer_G, args)

    encoder = Encoder(args.z_dim).cuda()
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=2e-4)

    print('  + Number of params of classifier: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('  + Number of params of generator: {}'.format(sum([p.data.nelement() for p in G.parameters()])))
    print('  + Number of params of generator: {}'.format(sum([p.data.nelement() for p in encoder.parameters()])))

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch, G, optimizer_G, encoder, optimizer_encoder)
        scheduler_G.step()

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch > 70:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))
            torch.save(G.state_dict(),
                       os.path.join(model_dir, 'generator-epoch{}.pt'.format(epoch)))
            torch.save(optimizer_G.state_dict(),
                       os.path.join(model_dir, 'optG_epoch{}.tar'.format(epoch)))
            torch.save(encoder.state_dict(),
                       os.path.join(model_dir, 'encoder-epoch{}.pt'.format(epoch)))
            torch.save(optimizer_encoder.state_dict(),
                       os.path.join(model_dir, 'opt-encoder_epoch{}.tar'.format(epoch)))

if __name__ == '__main__':
    main()
