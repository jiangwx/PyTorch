import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader,Dataset
from PIL import Image

import math
import time
import numpy as np
from model import darknet


def get_layer_fm_index(index, conv_cfg):
    layer_index = 0
    fm_index = 0
    while (index >= conv_cfg[layer_index]):
        index = index - conv_cfg[layer_index]
        layer_index = layer_index + 1
    fm_index = index
    return layer_index, fm_index

def get_index(layer_index, fm_index, conv_cfg):
    index = 0
    for i in range(layer_index):
        index = index + conv_cfg[i]
    index = index + fm_index
    return index

def generate_pruned_cfg(loss_rank, cfg, prune_cnt):
    conv_cfg = []  # store the fm cnt of each conv layer
    pruned_layer_cfg = [[]]  # store the remain fm index of each conv layer
    pruned_model_cfg = cfg  # store pruned cfg to generate new model

    for l in cfg:
        if l != 'M':
            conv_cfg.append(l[0])

    for i in range(len(cfg) - cfg.count('M') - 1):
        pruned_layer_cfg.append([])

    fm_cnt = len(loss_rank) - prune_cnt  # remained fm count

    for i in range(fm_cnt):
        layer_index, fm_index = get_layer_fm_index(loss_rank[i], conv_cfg)
        pruned_layer_cfg[layer_index].append(fm_index)

    layer_index = 0
    for i in range(len(cfg)):
        if cfg[i][0] != 'M':
            pruned_model_cfg[i][0] = len(pruned_layer_cfg[layer_index])
            layer_index += 1
    return pruned_model_cfg, pruned_layer_cfg

def pruned_model_init(model, pruned_model, pruned_layer_cfg):

    conv_layer_index = 0

    for layer_index, pruned_layer in pruned_model.feature._modules.items():

        _, layer = model.feature._modules.items()[int(layer_index)]

        if isinstance(pruned_layer, nn.Conv2d):

            if (conv_layer_index == 0):
                weight = layer.weight.data.cpu().numpy()
                pruned_weight = layer.weight.data.cpu().numpy()[:len(pruned_layer_cfg[conv_layer_index])]
                print pruned_weight.shape

                out_channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    pruned_weight[out_channel] = weight[i]
                    out_channel = out_channel + 1

                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

            else:
                weight = layer.weight.data.cpu().numpy()
                pruned_weight = layer.weight.data.cpu().numpy()[:len(pruned_layer_cfg[conv_layer_index]), :len(pruned_layer_cfg[conv_layer_index - 1])]
                print pruned_weight.shape

                in_channel = 0
                out_channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    for j in pruned_layer_cfg[conv_layer_index - 1]:
                        pruned_weight[out_channel, in_channel] = weight[i, j]
                        in_channel = in_channel + 1
                    in_channel = 0
                    out_channel = out_channel + 1

                pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

        elif isinstance(pruned_layer, nn.BatchNorm2d):

            if (conv_layer_index < 18):

                channel = 0
                for i in pruned_layer_cfg[conv_layer_index]:
                    pruned_layer.weight.data[channel] = layer.weight.data.cpu()[i].clone()
                    pruned_layer.bias.data[channel] = layer.bias.data.cpu()[i].clone()
                    pruned_layer.running_mean.data[channel] = layer.running_mean.data.cpu()[i].clone()
                    pruned_layer.running_var.data[channel] = layer.running_var.data.cpu()[i].clone()
                    channel = channel + 1

            conv_layer_index += 1

    for layer_index, pruned_layer in pruned_model.classifier._modules.items():

        _, layer = model.classifier._modules.items()[int(layer_index)]
        
        if isinstance(pruned_layer, nn.Conv2d):
            weight = layer.weight.data.cpu().numpy()
            pruned_weight = layer.weight.data.cpu().numpy()[:, :len(pruned_layer_cfg[-1])]
            print pruned_weight.shape

            in_channel = 0
            for i in pruned_layer_cfg[conv_layer_index - 1]:
                pruned_weight[:, in_channel] = weight[:, i]
                in_channel = in_channel + 1

            pruned_layer.weight.data = torch.from_numpy(pruned_weight).cuda()

def train(model, data_loader, loss_func, optimizer):
    model.train()
    correct = 0
    train_loss = 0.0

    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss.backward()
        optimizer.step()

    accuracy = 100. * float(correct) / float(len(data_loader.dataset))
    train_loss /= float(len(data_loader))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(train_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, train_loss

def test(model, data_loader, loss_func):
    model.eval()
    correct = 0
    loss = 0.0

    for inputs, labels in data_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss += loss_func(outputs, labels).item()

    accuracy = 100. * float(correct) / float(len(data_loader.dataset))
    loss /= float(len(data_loader))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, loss

def poly(base_lr, power, total_epoch, now_epoch):
    return base_lr * (1 - math.pow(float(now_epoch) / float(total_epoch), power))
   

train_data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_data_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root='/media/lulugay/PC/CCCV-30/train_set', transform=train_data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
test_dataset = torchvision.datasets.ImageFolder(root='/media/lulugay/PC/CCCV-30/test_set', transform=test_data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=12)

loss = np.random.random(7200)
loss_rank = np.argsort(loss)
cfg=[[32,3],'M',[64,3],'M',[128,3],[64,1],[128,3],'M',[256,3],[128,1],[256,3],'M',[512,3],[256,1],[512,3],[256,1],[512,3],'M',[1024,3],[512,1],[1024,3],[512,1],[1024,3]]
pruned_model_cfg,pruned_layer_cfg = generate_pruned_cfg(loss_rank, cfg, 3600)
print pruned_model_cfg
model = darknet()
model.cuda()
print model
pruned_model = darknet(cfg=pruned_model_cfg)
pruned_model.cuda()
print pruned_model

model.load_state_dict(torch.load('75.7.pkl'))
pruned_model_init(model, pruned_model, pruned_layer_cfg)
loss_func = nn.CrossEntropyLoss()
test(pruned_model,test_loader,loss_func)

start_epoch = 0
total_epoch = 120

history_score=np.zeros((total_epoch + 1,4))

for epoch in range(start_epoch, total_epoch):
    start = time.time()
    print('epoch%d...'%epoch)
    lr = poly(0.05, 4, total_epoch, epoch)
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    train_accuracy, train_loss = train(pruned_model,train_loader,loss_func,optimizer)
    test_accuracy, test_loss = test(pruned_model, test_loader, loss_func)

    torch.save(pruned_model.state_dict(), 'check_point.pkl')
    if test_accuracy > max(history_score[:,2]):
        torch.save(pruned_model.state_dict(), 'best.pkl')

    history_score[epoch][0] = train_accuracy
    history_score[epoch][1] = train_loss
    history_score[epoch][2] = test_accuracy
    history_score[epoch][3] = test_loss

    print('epoch%d time %.4fs\n' % (epoch,time.time()-start))