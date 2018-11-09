import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision
from torch.utils.data import DataLoader,Dataset
from PIL import Image

import math
import time

class DarkNet(nn.Module):
    def __init__(self):

        super(DarkNet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),\
                                   nn.BatchNorm2d(32),nn.LeakyReLU(0.1))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),\
                                    nn.BatchNorm2d(64),nn.LeakyReLU(0.1))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),\
                                   nn.BatchNorm2d(128),nn.LeakyReLU(0.1))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),\
                                   nn.BatchNorm2d(64),nn.LeakyReLU(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1, bias=False),\
                                   nn.BatchNorm2d(128),nn.LeakyReLU(0.1))
        self.pool5 = nn.MaxPool2d(2)
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),\
                                   nn.BatchNorm2d(256), nn.LeakyReLU(0.1))
        self.conv7= nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),\
                                   nn.BatchNorm2d(128), nn.LeakyReLU(0.1))
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),\
                                   nn.BatchNorm2d(256), nn.LeakyReLU(0.1))
        self.pool8 = nn.MaxPool2d(2)
        self.conv9 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),\
                                   nn.BatchNorm2d(512), nn.LeakyReLU(0.1))
        self.conv10 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),\
                                    nn.BatchNorm2d(256), nn.LeakyReLU(0.1))
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),\
                                    nn.BatchNorm2d(512), nn.LeakyReLU(0.1))
        self.conv12 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),\
                                    nn.BatchNorm2d(256), nn.LeakyReLU(0.1))
        self.conv13 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),\
                                    nn.BatchNorm2d(512), nn.LeakyReLU(0.1))
        self.pool13 = nn.MaxPool2d(2)
        self.conv14 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),\
                                    nn.BatchNorm2d(1024), nn.LeakyReLU(0.1))
        self.conv15 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),\
                                    nn.BatchNorm2d(512), nn.LeakyReLU(0.1))
        self.conv16 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),\
                                    nn.BatchNorm2d(1024), nn.LeakyReLU(0.1))
        self.conv17 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),\
                                    nn.BatchNorm2d(512), nn.LeakyReLU(0.1))
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool18 = nn.AvgPool2d(7)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.pool18(x)
        x = x.view(x.size(0), -1)
        return x
		
data_transform = transforms.Compose([
        transforms.RandomResizedCrop (224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

train_dataset = torchvision.datasets.ImageFolder(root='/media/lulugay/PC/CCCV/',transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle=True, num_workers=12)
 
val_dataset = torchvision.datasets.ImageFolder(root='/media/lulugay/PC/CCCV/', transform=data_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle=True, num_workers=12)

model = DarkNet()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1):
	batch_size_start = time.time()
	running_loss = 0.0
	for i, (inputs, labels) in enumerate(train_loader):
		inputs = inputs.cuda()
		labels = labels.cuda()
		inputs = Variable(inputs)
		lables = Variable(labels)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_func(outputs, labels)        #交叉熵
		loss.backward()
		optimizer.step()                          #更新权重
		running_loss += loss.data[0]
 
print('Epoch [%d/%d], Loss: %.4f,need time %.4f' % (epoch + 1, num_epochs, running_loss / (4000 / batch_size), time.time() - batch_size_start))
