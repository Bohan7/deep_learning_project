# -*- coding: utf-8 -*-
"""
Created on Sun May 23 00:02:07 2021

@author: Bohan
"""
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

# No weight sharing. No auxiliary loss.
class CNN(nn.Module):
     def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=64, output_fc=256, use_bn=True):
        super(CNN,self).__init__()
        
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3)
        self.active = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(out_channels_2*5*5, output_fc)
        self.fc2 = nn.Linear(output_fc, 2)

        
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels_1)
            self.bn2 = nn.BatchNorm2d(out_channels_2)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
             

        
     def forward(self, x):
        out = self.bn1(self.active(self.conv1(x)))
        out = self.bn2(self.active(self.conv2(out)))
        out = self.pool(out)
        out = out.view(out.size()[0], -1)
        out = self.active(self.fc1(out))
        out = self.fc2(out)
        
        return out

# Only weight sharing. 
class CNN_WS(nn.Module): 
    def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=64, output_fc1=50, output_fc2=25, use_bn=True):
        super(CNN_WS,self).__init__()
        
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels // 2, out_channels_1, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3)
        self.active = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(out_channels_2*5*5, output_fc1)
        self.fc2 = nn.Linear(output_fc1, 10)
        self.fc3 = nn.Linear(20, output_fc2)
        self.fc4 = nn.Linear(output_fc2, 2)

        
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels_1)
            self.bn2 = nn.BatchNorm2d(out_channels_2)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
   
    def one_branch(self, x):
        x = self.bn1(self.active(self.conv1(x)))
        x = self.bn2(self.active(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.active(self.fc1(x))
        x = self.fc2(x)

        return x
        
        
    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.reshape(-1, 1, 14, 14)
        x2 = x2.reshape(-1, 1, 14, 14)
        
        x1_class = self.one_branch(x1)
        x2_class = self.one_branch(x2)
        
        out = torch.cat((x1_class, x2_class), dim=1)
        out = self.active(self.fc3(out))
        out = self.fc4(out)
        return out, (x1_class, x2_class)

# No weight sharing. No auxiliary loss.
class Resnetblock(nn.Module):
     def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=32, output_fc=256, kernel_size=3, use_bn=True):
        super(Resnetblock,self).__init__()
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        self.active = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(out_channels_2, output_fc)
        self.fc2 = nn.Linear(output_fc, 2)
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_2, kernel_size=1),
                nn.BatchNorm2d(out_channels_2),
                )

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels_1)
            self.bn2 = nn.BatchNorm2d(out_channels_2)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
             

        
     def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.active(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.active(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.active(self.fc1(out))
        out = self.fc2(out)
        
        return out


# Only weight sharing. 
class Resnet_WS(nn.Module): 
    def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=64, output_fc=256, use_bn=True):
        super(CNN_WS,self).__init__()
        
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels // 2, out_channels_1, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3)
        self.active = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(out_channels_2*5*5, output_fc)
        self.fc2 = nn.Linear(output_fc, 10)
        self.fc3 = nn.Linear(20, 2)

        
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels_1)
            self.bn2 = nn.BatchNorm2d(out_channels_2)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
   
    def one_branch_feature(self, x):
        x = self.bn1(self.active(self.conv1(x)))
        x = self.bn2(self.active(self.conv2(x)))
        x = self.pool(x)
        
        return x
    
    def one_branch_classify(self, x):
        x = x.view(x.size()[0], -1)
        x = self.active(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
        
    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.reshape(-1, 1, 14, 14)
        x2 = x2.reshape(-1, 1, 14, 14)
        
        x1_feature = self.one_branch_feature(x1)
        x2_feature = self.one_branch_feature(x2)
        x1_class = self.one_branch_classify(x1_feature)
        x2_class = self.one_branch_classify(x2_feature)
        
        out = torch.cat((x1_class, x2_class), dim=1)
        out = self.fc3(out)
        return out, (x1_class, x2_class)    
        

        
