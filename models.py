## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)

        # max pooling layer with size 2 and stride 2
        self.pool1 = nn.MaxPool2d(2, 2)

        # 32 input image channel (grayscale), 64 output channels/feature maps, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3)

        # max pooling layer with size 2 and stride 2
        self.pool2 = nn.MaxPool2d(2, 2)

        # 64 input image channel (grayscale), 128 output channels/feature maps, 2x3 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 3)

        # max pooling layer with size 2 and stride 2
        self.pool3 = nn.MaxPool2d(2, 2)

        # 128 input image channel (grayscale), 256 output channels/feature maps, 3x3 square convolution kernel
        self.conv4 = nn.Conv2d(128, 256, 3)

        # max pooling layer with size 2 and stride 2
        self.pool4 = nn.MaxPool2d(2, 2)

        # 256 input image channel (grayscale), 512 output channels/feature maps, 1x1 square convolution kernel
        self.conv5 = nn.Conv2d(256, 512, 1)

        # max pooling layer with size 2 and stride 2
        self.pool5 = nn.MaxPool2d(2, 2)

        # dropout layer 1
        self.dropout1 = nn.Dropout(0.1)

        # dropout layer 2
        self.dropout2 = nn.Dropout(0.2)

        # dropout layer 3
        self.dropout3 = nn.Dropout(0.3)

        # dropout layer 4
        self.dropout4 = nn.Dropout(0.4)

        # conv batch normalization layer
        # self.conv2_norm = nn.BatchNorm2d(32)

        # fc batch normalization layer
        # self.fc2_norm = nn.BatchNorm1d(1024)

        # fully connected layer 1
        # the size of the first layer is calculated as follows:
        # ((224-5)+1)/2 = 110
        # ((110-5)+1)/2 = 53
        # ((53-3)+1)/2 = 25.5 ≈ 25
        # ((25-3)+1)/2 = 11.5 ≈  11
        # ((11-1)+1)/2 = 5.5 ≈  5

        self.fc1 = nn.Linear(6 * 6 * 512, 1024)

        # fully connected layer 2
        self.fc2 = nn.Linear(1024, 1024)

        # fully connected layer 3
        self.fc3 = nn.Linear(1024, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))

        x = self.dropout1(x)

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.dropout2(x)

        x = self.pool3(F.relu(self.conv3(x)))

        x = self.dropout3(x)

        x = self.pool4(F.relu(self.conv4(x)))

        x = self.dropout4(x)

        x = self.pool5(F.relu(self.conv5(x)))

        x = self.dropout2(x)

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.dropout3(x)

        x = F.relu(self.fc2(x))

        x = self.dropout4(x)

        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
