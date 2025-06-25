import torch
import torch.nn as nn
import torchvision.models as models
import torchvision


class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        vgg = models.vgg19(True)
        for pa in vgg.parameters():
            pa.requires_grad = False
        self.vgg = vgg.features[:16]
        self.vgg = self.vgg.to(device)

    def forward(self, x):
        out = self.vgg(x)
        return out

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

class ContentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg19 = VGG(device)

    def forward(self, fake, real):
        feature_fake = self.vgg19(fake)
        feature_real = self.vgg19(real)
        loss = self.mse(feature_fake, feature_real)
        return loss


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        ad_x = torch.ones_like(x).to('cuda')
        loss = self.bce(ad_x,x)
        return loss
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
class PerceptualLoss_1(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg_loss = ContentLoss(device)
        self.adversarial = AdversarialLoss()
        self.mse = nn.MSELoss()
        # self.mse = CharbonnierLoss()

    def forward(self, fake, real):
        vgg_loss = self.vgg_loss(fake, real)
        # adversarial_loss = self.adversarial(x)
        # mse_loss = self.mse(fake,real)
        # return mse_loss + 1e-3 * adversarial_loss
        return vgg_loss

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg_loss = ContentLoss(device)
        self.adversarial = AdversarialLoss()
        # self.mse = nn.MSELoss()
        self.mse = CharbonnierLoss()

    def forward(self, fake, real):
        # vgg_loss = self.vgg_loss(fake, real)
        # adversarial_loss = self.adversarial(x)
        mse_loss = self.mse(fake,real)
        # return mse_loss + 1e-3 * adversarial_loss
        return mse_loss

class RegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, 1:x.shape[2], :x.shape[3]-1]
        )
        b = torch.square(
            x[:, :, :x.shape[2]-1, :x.shape[3]-1] - x[:, :, :x.shape[2]-1, 1:x.shape[3]]
        )
        loss = torch.sum(torch.pow(a+b, 1.25))
        return loss


if __name__ == '__main__':
    # # c = ContentLoss()
    # f1 = torch.rand([1, 3, 64, 64])
    # r1 = torch.rand([1, 3, 64, 64])
    # # print(c(f1, r1))
    # # ad = AdversarialLoss()
    # i = torch.rand([1, 2, 1, 1])
    # # print(ad(i))
    # p = PerceptualLoss()
    # print(p(f1, r1, i))
    img = torch.rand([1, 3, 64, 64])
    r = RegularizationLoss()
    print(r(img))










