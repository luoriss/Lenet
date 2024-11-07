#1、SENet配置了两次激活，每用一次全链接，就要用一次激活。这个值得学习。这个是基本
#2、b, c, _, _ = x.size() 和  y = self.avg_pool(x).view(b, c) 都是值得学习的地方。

#coding: UTF-8
import scipy.io as io
import torch
from torch import nn
import numpy as np

from icecream import ic


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        assert channel > reduction, "Make sure your input channel bigger than reduction which equals to {}".format(reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #[batchsize, channels, 1, 1]
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                # nn.Sigmoid()
                nn.Softmax()
        )

    def forward(self, x):
        b, c, _, _ = x.size()    #[batchsize, channels]
        y = self.avg_pool(x).view(b, c)   #[batchsize, channels]
        y = self.fc(y).view(b, c, 1, 1)   #[batchsize, channels, 1, 1]
        y_mean = y.mean(axis=1, keepdim=False)  #[batchsize, 1, 1]

        ic(y_mean.shape)
        
        print('均值为：')
        print(y_mean.squeeze())
        result1 = np.array(y_mean.detach())

        print('方差为：')
        y_var = y.var(axis=1, keepdim=False)
        print(y_var.squeeze())
        ic(y_var.shape)
        return x * y

if __name__ == '__main__':
    input = torch.rand(2,64,128,128)
    b, c, _, _ = input.size()  #[batchsize, channels, weight, height]
    senet = SELayer(64)

    y = senet(input)

    print(y.shape)


class SELayer0(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer0,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


