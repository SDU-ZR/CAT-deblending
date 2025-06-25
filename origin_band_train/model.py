import math
import torch
import torch.nn as nn
import torchvision
import numpy as np

class ConvolutionBlock(nn.Module):
    """
    卷积模块，由卷积层，BN归一化层，激活层构成
    """
    def __init__(self,in_channels=3,out_channels=64,kernel_size=3,stride=1,batch_norm=False,activation=None):
        """
        :参数 in_channels:输入通道数
        """
        super(ConvolutionBlock,self).__init__()
        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu','leakyrelu','tanh'}
        layers = list()
        layers.append(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=kernel_size//2))
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        #合并层
        self.conv_block = nn.Sequential(*layers)
    def forward(self,input):
        output = self.conv_block(input)
        return output

class ResidualBlock(nn.Module):
    """
    残差模块，包含两个卷积模块和跳连
    """
    def __init__(self,kernel_size=3,n_channels=64):
        """
        :参数 kernel_size :核大小
        :参数 n_channels :输入核输出通道数(由于是ResNet网络，需要做跳连，因此输入与输出通道一致）
        """
        super(ResidualBlock,self).__init__()
        # 第一个卷积块
        self.conv_block1 = ConvolutionBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # 第二个卷积块
        self.conv_block2 = ConvolutionBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)
    def forward(self,input):
        """
        前向传播
        :参数 input: 输入图像集，张量表示，大小为（N,n_channels,w,h)
        :返回：输出图像集，张量表示，大小为(N，n_channels,w,h)
        """
        residual = input
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = output+residual

        return output

class Generator(nn.Module):
    """
    生成器模型
    """
    def __init__(self,large_kernel_size=9,small_kernel_size=3,n_channels=64,num_root_blocks=10, num_branch_blocks=6):
        """
        :param large_kernel_size:
        :param small_kernel_size:
        :param n_channels:
        :param num_root_blocks:
        :param num_branch_blocks:
        """
        super(Generator, self).__init__()

        """
        Root
        """
        self.conv_blocks1 = ConvolutionBlock(in_channels=3,out_channels=n_channels,kernel_size=large_kernel_size,batch_norm=False,activation='PReLu')

        #一系列的残差模块
        self.root_residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(num_root_blocks)])

        self.conv_blocks2 = ConvolutionBlock(in_channels=n_channels,out_channels=n_channels,kernel_size=small_kernel_size,
                                             batch_norm=True,
                                             activation=None)
        """
        Branch One
        """
        self.branch_one_residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(num_branch_blocks)])

        self.conv_block3 = ConvolutionBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                            batch_norm=False, activation='Tanh')


    def forward(self,x):
        """
        前向传播
        :param x: 预混合图像
        :return: 两张解混合图像
        """
        x = self.conv_blocks1(x)
        residual = x
        output = self.root_residual_blocks(x)
        output = residual + output
        output = self.conv_blocks2(output)
        # 带入分支one
        y = self.branch_one_residual_blocks(output)
        y = self.conv_block3(y)


        return y



class TruncatedVGG19(nn.Module):
    """
    truncated VGG19网络，用于计算VGG特征空间的MSE损失
    """

    def __init__(self, i, j):
        """
        :参数 i: 第 i 个池化层
        :参数 j: 第 j 个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG模型
        vgg19 = torchvision.models.vgg19(pretrained=True)  # C:\Users\Administrator/.cache\torch\checkpoints\vgg19-dcbb9e9d.pth

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # 迭代搜索
        for layer in vgg19.features.children():
            truncate_at += 1

            # 统计
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 截断位置在第(i-1)个池化层之后（第 i 个池化层之前）的第 j 个卷积层
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)

        # 截取网络
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        前向传播
        参数 input: 高清原始图或超分重建图，张量表示，大小为 (N, 3, w * scaling factor, h * scaling factor)
        返回: VGG19特征图，张量表示，大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
"""
class Discriminator(nn.Module):

    def __init__(self,kernel_size=3,n_channels=64,n_blocks=8,fc_size=1024):

        super(Discriminator,self).__init__()
        in_channels = 3
        #卷积系列，论文中的SRGAN
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvolutionBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 == 0 else 2, batch_norm=i != 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)
        # 固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()

    def forward(self,imgs):
        self.fc2 = nn.Linear(1024,1)
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)

        logit = self.fc2(output)
        logit = self.sig(logit)
        return logit
"""

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(torch.unsqueeze(self.net(x).view(batch_size),1))

if __name__ =="__main__":

    x = torch.rand(8, 3, 100, 100)
    model = Generator()
    A,B = model(x)
    model_B = Discriminator()
    C = model_B(B).to(torch.float32)
    print(B.shape)
    print(C.shape)





