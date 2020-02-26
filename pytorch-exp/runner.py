#!/usr/local/bin/python
from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from nets.BaseNet import BaseNet
from nets.ShortCircuitNet import ShortCircuitNet
from nets.Conv1Net import Conv1Net
from nets.ConvStackerNet import ConvStackerNet
from nets.layers.convs.ConvLayer import ConvLayer

# this framework was taken from https://github.com/pytorch/examples/blob/master/mnist/main.py

def train(args, model, device, train_loader, optimizer, epoch, write_file = None):
    model.train()
    correct = 0
    total_training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()

        # calculating stats
        loss = loss.item()
        total_training_loss += loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        if write_file:
            write_file.write('1\n' if pred[0].item() == target.item() else '0\n')
        # print(output)
        # print(pred)
        # print(target)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0 and args.verbose_log:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss / args.batch_size))
    
    total_training_loss /= len(train_loader.dataset)
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        total_training_loss, correct, len(train_loader.dataset), 
        100. * correct / len(train_loader.dataset)))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', default='mnist',
                        help='dataset to run on')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--verbose-log', action='store_true', default=False, 
                        help='set true to see stats for each log-interval')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--run-id', default='',
                        help='save file identifier for runs')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
        mnist_wrapper(device, args, train_loader, test_loader)

    elif args.dataset == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        
        cifar_wrapper(device, args, train_loader, test_loader)
    
def mnist_wrapper(device, args, train_loader, test_loader):
    print('------Full Model------')
    full_model = BaseNet(1, 32, 64, 9216, 128).to(device)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print('--Epoch ' + str(epoch) + '--')

        start = time.time()
        train(args, full_model, device, train_loader, optimizer, epoch)
        print('Training time: ' + str(time.time() - start))

        start = time.time()
        test(args, full_model, device, test_loader)
        print('Testing time: ' + str(time.time() - start))
    
    if args.save_model:
        torch.save(full_model.state_dict(), "./saved_models/mnist-full_model-" + args.run_id + ".pt")

    print('------Conv 1 Model------')
    conv1_model = Conv1Net(1, 32, [5408, 128, 10]).to(device)
    optimizer = torch.optim.Adam(conv1_model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        print('--Epoch ' + str(epoch) + '--')

        start = time.time()
        train(args, conv1_model, device, train_loader, optimizer, epoch)
        print('Training time: ' + str(time.time() - start))        
        
        start = time.time()
        test(args, conv1_model, device, test_loader)
        print('Testing time: ' + str(time.time() - start))
    
    if args.save_model:
        torch.save(conv1_model.state_dict(), "./saved_models/mnist-conv1_model-" + args.run_id + ".pt")

    # save the conv weights / bias
    torch.save({'conv.weight': conv1_model.state_dict()['conv1.conv.weight'], 
                'conv.bias': conv1_model.state_dict()['conv1.conv.bias']}, './tmp/mnist-conv1_weights.pt')
    saved_weights = torch.load('./tmp/mnist-conv1_weights.pt')

    print('------Conv 2 Model------')
    conv1_new = ConvLayer(1, 32)
    conv1_new.load_state_dict(saved_weights)

    for param in conv1_new.parameters():
        param.requires_grad = False

    conv2_model = ConvStackerNet([conv1_new], 32, 64, [9216, 128, 10]).to(device)
    optimizer = torch.optim.Adam(conv2_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print('--Epoch ' + str(epoch) + '--')

        start = time.time()
        train(args, conv2_model, device, train_loader, optimizer, epoch)
        print('Training time: ' + str(time.time() - start))        
        
        start = time.time()
        test(args, conv2_model, device, test_loader)
        print('Testing time: ' + str(time.time() - start))

    if args.save_model:
        torch.save(conv2_model.state_dict(), "./saved_models/mnist-conv2_model-" + args.run_id + ".pt")

def cifar_wrapper(device, args, train_loader, test_loader):
    ## Short Circuit Model ###
    write_file = open('execution.txt', 'w')
    sc_model = ShortCircuitNet(0.15, write_file=write_file).to(device)
    optimizer = torch.optim.Adam(sc_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        write_file.write('---\n')
        print('--Epoch ' + str(epoch) + '--')

        start = time.time()
        train(args, sc_model, device, train_loader, optimizer, epoch, write_file=write_file)
        print('Training time: ' + str(time.time() - start))

        start = time.time()
        test(args, sc_model, device, test_loader)
        print('Testing time: ' + str(time.time() - start))
    
    if args.save_model:
        torch.save(sc_model.state_dict(), "./saved_models/sc-" + args.run_id + ".pt")

    # print('------Full Model------')
    # full_model = BaseNet(3, 32, 64, 12544, 256).to(device)
    # optimizer = torch.optim.Adam(full_model.parameters(), lr=args.lr)

    # for epoch in range(1, args.epochs + 1):
    #     print('--Epoch ' + str(epoch) + '--')

    #     start = time.time()
    #     train(args, full_model, device, train_loader, optimizer, epoch)
    #     print('Training time: ' + str(time.time() - start))

    #     start = time.time()
    #     test(args, full_model, device, test_loader)
    #     print('Testing time: ' + str(time.time() - start))
    
    # if args.save_model:
    #     torch.save(full_model.state_dict(), "./saved_models/cifar-full_model-" + args.run_id + ".pt")

    ########### Conv1 Model setup ###########
    # print('------Conv 1 Model------')
    # conv1_model = Conv1Net(3, 32, [7200, 1024, 256, 10]).to(device)
    # optimizer = torch.optim.Adam(conv1_model.parameters(), lr=args.lr)
    # for epoch in range(1, args.epochs + 1):
    #     print('--Epoch ' + str(epoch) + '--')

    #     start = time.time()
    #     train(args, conv1_model, device, train_loader, optimizer, epoch)
    #     print('Training time: ' + str(time.time() - start))        
        
    #     start = time.time()
    #     test(args, conv1_model, device, test_loader)
    #     print('Testing time: ' + str(time.time() - start))
    
    # if args.save_model:
    #     torch.save(conv1_model.state_dict(), "./saved_models/cifar-conv1_model-" + args.run_id + ".pt")


    # ########### Conv2 Model setup ###########
    # # save the conv weights / bias
    # torch.save({'conv.weight': conv1_model.state_dict()['conv1.conv.weight'], 
    #             'conv.bias': conv1_model.state_dict()['conv1.conv.bias']}, './tmp/cifar-conv1_weights.pt')
    # saved_weights = torch.load('./tmp/cifar-conv1_weights.pt')

    # print('------Conv 2 Model------') # first layer pretrained
    # conv1_new = ConvLayer(3, 32)
    # conv1_new.load_state_dict(saved_weights)

    # for param in conv1_new.parameters():
    #     param.requires_grad = False

    # conv2_model = ConvStackerNet([conv1_new], 32, 64, [12544, 1024, 256, 10]).to(device)
    # optimizer = torch.optim.Adam(conv2_model.parameters(), lr=args.lr)

    # for epoch in range(1, args.epochs + 1):
    #     print('--Epoch ' + str(epoch) + '--')

    #     start = time.time()
    #     train(args, conv2_model, device, train_loader, optimizer, epoch)
    #     print('Training time: ' + str(time.time() - start))        
        
    #     start = time.time()
    #     test(args, conv2_model, device, test_loader)
    #     print('Testing time: ' + str(time.time() - start))

    # if args.save_model:
    #     torch.save(conv2_model.state_dict(), "./saved_models/cifar-conv2_model-" + args.run_id + ".pt")

    ########### Conv3 Model setup ###########
    # save the conv1 weights / bias
    # torch.save({'conv.weight': conv2_model.state_dict()['prev_layer0.conv.weight'], 
    #             'conv.bias': conv2_model.state_dict()['prev_layer0.conv.bias']}, './tmp/cifar-conv1_weights.pt')
    # conv1_saved_weights = torch.load('./tmp/cifar-conv1_weights.pt')

    # save the conv2 weights / bias
    # torch.save({'conv.weight': conv2_model.state_dict()['last_conv.conv.weight'], 
    #             'conv.bias': conv2_model.state_dict()['last_conv.conv.bias']}, './tmp/cifar-conv2_weights.pt')
    # conv2_saved_weights = torch.load('./tmp/cifar-conv2_weights.pt')

    # print('------Conv 3 Model------') # first 2 layers pretrained
    # conv1_new = ConvLayer(3, 32)
    # conv1_new.load_state_dict(conv1_saved_weights)

    # for param in conv1_new.parameters():
    #     param.requires_grad = False

    # conv2_new = ConvLayer(32, 64)
    # conv2_new.load_state_dict(conv2_saved_weights)

    # for param in conv2_new.parameters():
    #     param.requires_grad = False

    # conv3_model = ConvStackerNet([conv1_new, conv2_new], 64, 128, [21632, 1024, 256, 10]).to(device)
    # optimizer = torch.optim.Adam(conv3_model.parameters(), lr=args.lr)

    # for epoch in range(1, args.epochs + 1):
    #     print('--Epoch ' + str(epoch) + '--')

    #     start = time.time()
    #     train(args, conv3_model, device, train_loader, optimizer, epoch)
    #     print('Training time: ' + str(time.time() - start))        
        
    #     start = time.time()
    #     test(args, conv3_model, device, test_loader)
    #     print('Testing time: ' + str(time.time() - start))

    # if args.save_model:
    #     torch.save(conv3_model.state_dict(), "./saved_models/cifar-conv3_model-" + args.run_id + ".pt")

if __name__ == '__main__':
    main()