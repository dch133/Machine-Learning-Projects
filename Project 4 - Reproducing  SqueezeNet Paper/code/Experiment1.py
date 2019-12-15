# -*- coding: utf-8 -*-
"""conv7x7; maxpool2x2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BNv__C_Md17vj-Mn4Nd0Vc1qHs2ge6BJ
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

import matplotlib.pyplot as plt
import numpy as np

epochs = 20

"""# Load data and transform"""

transform = transforms.Compose(
    [transforms.RandomRotation(30),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train= True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train= False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4,
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""# SqueezeNet for 32x32 CIFAR"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
            )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

def squeezenet():
    net = SqueezeNet()
    return net

"""## Load previous model"""

net  = squeezenet()

import torch.optim as optim

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

"""# Train the network"""

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# move model to device
net.to(device)

steps = 0

# Model training and validation
loss_Training = []
accuracy_Training = []
# loss_list_validation = []
accuracy_Validation = []

since = time.time()
for epoch in range(epochs):
    correct = 0
    total = 0
    running_loss = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # Move input and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Get predicted classes and number of correctly classified instances
        y_pred = torch.max(outputs.data, 1)[1]
        correct += float((y_pred == labels).sum())
        total += float(len(labels))


        loss.backward()
        optimizer.step()

        # Store loss value on training
        loss_Training.append(loss.item())

        #print statistics
        running_loss += loss.item()

        if i % 2000 == 1999: #print every 2000 mini-batches
            print ('[%d, %5d] avg loss: %.3f accuracy so far: %.3f' %
                   (epoch + 1, i + 1, running_loss / 2000,100*correct/total ))
            running_loss = 0

    # Store loss value on training

    # Calculate accuracy and store on training
    accuracy_training = 100*correct/total
    accuracy_Training.append(accuracy_training)
    
    correctV = 0
    totalV = 0
    with torch.no_grad():
      for dataV in testloader:
          imagesV, labelsV = dataV
          imagesV, labelsV = imagesV.to(device), labelsV.to(device)
          outputsV = net(imagesV)
          _, predictedV = torch.max(outputsV.data.cuda(), 1)
          totalV += labelsV.size(0)
          correctV += (predictedV == labelsV).sum().item()
    accuracy_validation = 100*correctV/totalV
    accuracy_Validation.append(accuracy_validation)


    print(f"Epoch : {epoch+1}     "
          f"Total Time : {time.time() - since}         "
          f"Training Accuracy : {accuracy_training}         "
          f"Validation Accuracy : {accuracy_validation}     "
          f"Training Loss : {loss.item()}     ")
      
    # TEMP: loss and accuracy plots
    f = plt.figure(2)
    plt.plot(range(1,len(accuracy_Training)+1), accuracy_Training)
    plt.plot(range(1,len(accuracy_Validation)+1), accuracy_Validation)
    plt.title('Accuracy vs number of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(('Training','Validation'), loc='upper left')
    f.show()
    f.savefig('Accuracy.jpg')
    plt.close(f)

    f = plt.figure(3)
    plt.plot(range(1,len(loss_Training)+1), loss_Training)
    # plt.plot(range(len(loss_list_validation)), loss_list_validation)
    plt.title('Training Loss vs number of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(('Training loss'), loc='upper left')
    f.show()
    f.savefig('Loss.jpg')
    plt.close(f)