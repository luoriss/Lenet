import torch
from torch import nn
from mmcv.cnn.utils import constant_init, kaiming_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)#用值val填充向量m[-1]
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']#用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])#融和的方式有两个
        assert len(fusions) > 0, 'at least one fusion should be used'#至少选择一种
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:#pool有两个形式，当是第一种形式的时候，执行下列的语句。
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#将执行一个1*1卷积，改变输出通道为1
            self.softmax = nn.Softmax(dim=2)#然后执行softmax
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)#如果不是第一种，那么执行
            #自适应平均池化函数（只需要给定输出特征图的大小就好，其中通道数前后不发生变化）
        if 'channel_add' in fusions:#如果融合方式选择的是通道相加，那么执行下列语句：
            self.channel_add_conv = nn.Sequential(#定义  通道相加融合方式 的 函数
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),#先变通道
                nn.LayerNorm([self.planes, 1, 1]),#归一化层：channel方向做归一化，算CHW的均值
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)#还原 原来的通道
            )
        else:#如果 fusion中没有  通道相加，则通道相加这项没有
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')#神经网络初始化kaiming分布
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()#传入   batch， C  H  W
        if self.pool == 'att':
            input_x = x# ([1, 256, 125, 88])[N, C, H, W]

            input_x = input_x.view(batch, channel, height * width)# [N, C, H * W]

            input_x = input_x.unsqueeze(1)# [N, 1, C, H * W]在第二个维度增加一个维度
            # 到这里只是对x的操作得到 input_x

            context_mask = self.conv_mask(x)# [N, 1, H, W]通道变为1

            context_mask = context_mask.view(batch, 1, height * width) # [N, 1, H * W]

            context_mask = self.softmax(context_mask) # [N, 1, H * W]

            context_mask = context_mask.unsqueeze(3)# [N, 1, H * W, 1]第四个维度加1

            #input_x [N, 1, C, H * W],,,context_mask [N, 1, H * W, 1]
            context = torch.matmul(input_x, context_mask)# [N, 1, C, 1]
            #torch.matmul是tensor的乘法

            context = context.view(batch, channel, 1, 1)# [N, C, 1, 1]

        else:#如果if self.pool 不等于 'att'方法， 则把输入x送入
            context = self.avg_pool(x)#平均池化操作，把输入[N, C, H, W]----[N, C, 1, 1]

        return context

    def forward(self, x):

        context = self.spatial_pool(x)# [N, C, 1, 1]把输入送入  前边定义的spatial_pool输入[N, C, 1, 1]

        if self.channel_mul_conv is not None:#如果是通道相乘，则执行下列语句

            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context)) # [N, C, 1, 1]
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:#如果是通道相加，则执行下列语句

            channel_add_term = self.channel_add_conv(context)# [N, C, 1, 1]
            out = out + channel_add_term

        return out
