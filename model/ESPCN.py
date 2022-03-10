import torch
import torch.nn as nn

class ESPCN(nn.Module):
#upscale_factor -> args
    def __init__(self, n_colors, scale): # n_colors颜色通道数 #scale放大倍数
        super(ESPCN, self).__init__()

        print("Creating ESPCN (x%d)" %scale)

        self.conv1 = nn.Conv2d(n_colors, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, n_colors * scale * scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale) # 将低分辨率图像变成高分辨率
        self.conv4 = nn.Conv2d(n_colors, n_colors, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x