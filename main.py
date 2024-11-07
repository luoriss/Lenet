#author = liuwei

# Python gzip模块提供了一种非常简单的方式来压缩和解压缩文件，
# 并以类似于GNU程序gzip和gunzip的方式工作。

# struct：Interpret strings as packed binary data.
# 具体作用就是用来处理字节流的，类似于c语言的struct.

# nn.functional是一个很常用的模块，nn中的大多数layer在functional中都有一个与之对应的函数。
# nn.functional中的函数与nn.Module()的区别是：

# nn.Module实现的层（layer）是一个特殊的类，都是由class Layer(nn.Module)定义，会自动提取可学习的参数
# nn.functional中的函数更像是纯函数,由def functional(input）定义，没有可学习的参数

from icecream import ic

import gzip, struct
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import math

# from SENet import SELayer

#读取数据的函数,先读取标签，再读取图片
def _read(image, label):
    minist_dir = 'data/'
    with gzip.open(minist_dir + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(minist_dir + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image, label 

#1、读取数据，标准化格式
def get_data():
    train_img, train_label = _read(
	    'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz')

    test_img, test_label = _read(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    return [train_img, train_label, test_img, test_label]


#定义lenet5
class LeNet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        #定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        #第二个卷积层，6个输入，16个输出，5*5的卷积filter 
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        #self.senet = SELayer(16,reduction=8)
        #最后是三个全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        

    def forward(self, x):
        '''前向传播函数'''
        #先卷积，然后调用relu激活函数，再最大值池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #第二次卷积+池化操作
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = self.senet(x)
        
        #重新塑形,将多维数据重新塑造为二维数据
        x = x.view(-1, self.num_flat_features(x))
        #print('size', x.size())
        #第一个全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        #，size的值为(16, 5, 5)，8是batch_size
        size = x.size()[1:]  #x.size（）返回的是一个元组(8, 16, 5, 5)，size=（16, 5, 5)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

#定义一些超参数
use_gpu = torch.cuda.is_available() #T  F
batch_size = 8
kwargs = {'num_workers': 2, 'pin_memory': True}                              #DataLoader的参数

#参数值初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()

#训练函数
def train(epoch):
    #调用前向传播
    print('Train Epoch= %d' %epoch)
    model.train()		
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()

        target = target.long()   #损失函数的要求
        
        #定义为Variable类型，能够调用autograd
        data, target = Variable(data), Variable(target)                      

        # 初始化时，要清空梯度
        optimizer.zero_grad()
        output = model(data)
        output = output.float()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() 
                                                            #相当于更新权重值
        if batch_idx % 100 == 0:
            # print('  batch_idx = %d' %batch_idx)
            # print('  loss = %f' %loss.data.item())

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100* batch_idx / len(train_loader), loss.data.item()))

#定义测试函数			   
def test():
    model.eval()                                                             #让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target = target.long() 

        output = model(data)
        output = output.float()
        #计算总的损失
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]                           #获得得分最高的类别
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

#1-1 获取数据
X, y, Xt, yt = get_data()

train_x, train_y = [torch.from_numpy(X.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(y.astype(int))]
test_x, test_y = [torch.from_numpy(Xt.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(yt.astype(int))]
print(type(train_y))
print(len(train_x))

#1-2 根据指定格式封装好数据和标签
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

#1-3 定义数据加载器（每次加载一部分，不然内存溢出）
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, **kwargs)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size, **kwargs)

#2-1 实例化网络
model = LeNet5()

if use_gpu:
    model = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')

#3-1 定义代价函数，使用交叉熵验证
criterion = nn.CrossEntropyLoss(size_average=False)  #每个小批次的损失将被相加

#3-2 直接调用优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

#3-3 调用参数初始化方法初始化网络中的所有参数
model.apply(weight_init)    #weight_init(model) 


# if __name__ == '__main__':的作用
# 一个python文件通常有两种使用方法，第一是作为脚本直接执行，第二是import到其他的python脚本中被调用（模块重用）执行。
# if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程，
# 在 if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，
# 而 import 到其他脚本中是不会被执行的。                                          #了解apply用法

if __name__ == '__main__':
#4-1 调用函数执行训练和测试
    for epoch in range(1):
        print('----------------start train-----------------')
        train(epoch)
        print('----------------end train-----------------')

        print('----------------start test-----------------')
        test()
        print('----------------end test-----------------')

    # ic(model)
