import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

class CNN(nn.Module):
     """
      CNN：Baseline model
     """

     def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=64, output_fc=30, use_bn=True):
        """
        :param input_channels: Number of input channels
        :param output_channels_1: Number of output channels for the first convolutional layer
        :param output_channels_2: Number of output channels for the second convolutional layer
        :param output_fc: Number of output channels of the full connected layer
        :param use_bn: whether use batch normalization
        """
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
 
class Siamese_net(nn.Module):
    """
    The model with weight sharing based on Siamese networks
    """ 
    def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=64, output_fc1=50, output_fc2=25, use_bn=True, version=1):
        """
        :param input_channels: Number of input channels
        :param output_channels_1: Number of output channels for the first convolutional layer
        :param output_channels_2: Number of output channels for the second convolutional layer
        :param output_fc_1: Number of output channels of the first full connected layer
        :param output_fc_2: Number of output channels of the third full connected layer
        :param use_bn: whether use batch normalization
        :param version: if version=1, output the predicted classes for each image of the input pairs, and the final target
                 if version=2, output the predicted classes for each image, and the final target is predicted by the predicted classes
        """
        super(Siamese_net,self).__init__()
        self.version = version
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
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
        """
        implement one branch forward of the siamese network
        """
        x = self.bn1(self.active(self.conv1(x)))
        x = self.bn2(self.active(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.active(self.fc1(x))
        x = self.fc2(x)

        return x
        
        
    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1] # split the two-channel input to two one-channel tensors
        x1 = x1.reshape(-1, 1, 14, 14)
        x2 = x2.reshape(-1, 1, 14, 14)
        
        x1_class = self.one_branch(x1)
        x2_class = self.one_branch(x2)
        
        if self.version == 1:
          out = torch.cat((x1_class, x2_class), dim=1)
          out = self.active(self.fc3(self.active(out)))
          out = self.fc4(out)
        elif self.version == 2:
          _, predicted_digit1 = torch.max(x1_class, 1)
          _, predicted_digit2 = torch.max(x2_class, 1)
          out = (predicted_digit1 <= predicted_digit2).float()

        return out, (x1_class, x2_class)

class Resnetblock(nn.Module):
     """
     The model based on Resnet
     """ 
     def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=32, output_fc=256, kernel_size=3, use_bn=True):
        """
        :param input_channels: Number of input channels
        :param output_channels_1: Number of output channels for the first convolutional layer
        :param output_channels_2: Number of output channels for the second convolutional layer
        :param output_fc: Number of output channels of the first full connected layer
        :param use_bn: whether use batch normalization
        """
        super(Resnetblock,self).__init__()
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        self.active = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(out_channels_2*7*7, output_fc)
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

class Resnetblock_WS(nn.Module):
     """
     The model based on Resnet with weight sharing
     """ 
     def __init__(self, in_channels=2, out_channels_1=32, out_channels_2=32, output_fc=256, kernel_size=3, use_bn=True):
        """
        :param input_channels: Number of input channels
        :param output_channels_1: Number of output channels for the first convolutional layer
        :param output_channels_2: Number of output channels for the second convolutional layer
        :param output_fc: Number of output channels of the first full connected layer
        :param kernel_size: The size of kernel used by convolutional layers
        :param use_bn: Whether use batch normalization
        """
        super(Resnetblock_WS,self).__init__()
        self.kernel_size = kernel_size
        self.use_bn = use_bn
        self.active = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels//2, out_channels_1, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(out_channels_2*7*7, output_fc)
        self.fc2 = nn.Linear(output_fc, 10)
        self.fc3 = nn.Linear(20, 2)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels//2, out_channels_2, kernel_size=1),
                nn.BatchNorm2d(out_channels_2),
                )

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels_1)
            self.bn2 = nn.BatchNorm2d(out_channels_2)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
             
     def one_branch(self, x):
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

        
     def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.reshape(-1, 1, 14, 14)
        x2 = x2.reshape(-1, 1, 14, 14)
        
        x1_class = self.one_branch(x1)
        x2_class = self.one_branch(x2)
        
        out = torch.cat((x1_class, x2_class), dim=1)
        out = self.fc3(self.active(out))

        return out, (x1_class, x2_class)


