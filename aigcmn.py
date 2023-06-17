import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import *

# 初始化模型
def get_model():
    latent_dim = 100
    condition_dim = 10
    img_shape = (1, 28, 28)
    generator = Generator(latent_dim, condition_dim, img_shape)
    state_dict = torch.load('cgan_generator_cuda.pth')
    generator.load_state_dict(state_dict)
    return generator


# 生成条件图片
def generate_image(condition, generator):
    with torch.no_grad():
        noise = torch.randn(1, latent_dim)
        condition = torch.zeros((1, 10)).scatter_(1, torch.tensor([[condition]]), 1)
        gen_img = generator(noise, condition)
        gen_img = gen_img.view(28, 28)
        gen_img = (gen_img + 1) / 2
        return gen_img.numpy()


def test():
    generator = get_model()
    generator.eval()

    flag = input("需要顺序输出0-9请输入：1 \n需要按要求输出指定数字请输入：0\n")
    if flag == '1':
        print("现在开始为您按序输出")
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                condition = i
                image = generate_image(condition, generator)
                axs[i, j].imshow(image, cmap='gray')
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.show()
    if flag == '0':
        while True:
            print("")
            print("如果想退出请输入-1")
            condition = int(input("请输入你想要输出的数字："))
            if condition == -1:
                return
            fig, axs = plt.subplots(1, 10, figsize=(10, 1))
            for j in range(10):
                image = generate_image(condition, generator)
                axs[j].imshow(image, cmap='gray')
                axs[j].axis('off')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    test()

