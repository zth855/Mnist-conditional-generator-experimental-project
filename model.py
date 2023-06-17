import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#超参数设置
latent_dim = 100
condition_dim = 10
img_shape = (1, 28, 28)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, img_shape):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), -1)
        img = self.model(x)
        img = img.view(img.size(0), *img_shape)
        return img

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, img_shape, condition_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))) + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, condition):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, condition), -1)
        validity = self.model(x)
        return validity


if __name__ == '__main__':
    pass