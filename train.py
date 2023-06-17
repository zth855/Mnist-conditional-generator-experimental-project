import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import * 
import time
import os

#超参数设置
latent_dim = 100
condition_dim = 10
img_shape = (1, 28, 28)
epochs = 200
batch_size = 32
lr = 0.0002
device = 'cuda'

def train():
    # 获取MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 初始化生成器和判别器
    generator = Generator(latent_dim, condition_dim, img_shape)
    discriminator = Discriminator(img_shape, condition_dim)

    # 设置优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # 设置损失函数
    adversarial_loss = nn.BCELoss()
    d_losses = []
    g_losses = []
    for epoch in range(epochs):
        single_train(epoch, generator, discriminator,optimizer_G, optimizer_D, train_loader, adversarial_loss, d_losses, g_losses)

    # 保存loss变化图
    save_dir="./results"
    cur = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # 保存d_loss
    label = 'd_loss lr: {}, batch_size: {}, epochs: {}'.format( lr, batch_size, epochs)
    plt.plot(d_losses,label=label)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, 'loss'+ cur +'.png'))
    # 保存g_loss
    label = 'g_loss lr: {}, batch_size: {}, epochs: {}'.format( lr, batch_size, epochs)
    plt.plot(g_losses,label=label)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, 'loss'+ cur +'.png'))
    
    # 保存模型
    model_save(generator)


#单步训练过程
def single_train(epoch, generator, discriminator,optimizer_G, optimizer_D, train_loader, adversarial_loss, d_losses, g_losses):
    # 单个epoch训练
    for i, (imgs, labels) in enumerate(train_loader):

        batch_size = imgs.size(0)

        # 真实图片的标签
        valid = torch.full((batch_size, 1), 1.0)
        # 生成图片的标签
        fake = torch.full((batch_size, 1), 0.0)

        # 训练判别器
       
        optimizer_D.zero_grad()

        # 真实图片
        real_imgs = imgs
        real_condition = torch.zeros((batch_size, 10)).scatter_(1, labels.view(-1, 1), 1)
        real_validity = discriminator(real_imgs, real_condition)
        real_loss = adversarial_loss(real_validity, valid)

        # 生成图片
        noise = torch.randn(batch_size, latent_dim)
        gen_labels = torch.randint(0, 10, (batch_size,)).long()
        gen_condition = torch.zeros((batch_size, 10)).scatter_(1, gen_labels.view(-1, 1), 1)
        gen_imgs = generator(noise, gen_condition)
        fake_validity = discriminator(gen_imgs.detach(), gen_condition)
        fake_loss = adversarial_loss(fake_validity, fake)

        # 计算判别器的损失
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器

        optimizer_G.zero_grad()

        gen_validity = discriminator(gen_imgs, gen_condition)
        g_loss = adversarial_loss(gen_validity, valid)
        g_loss.backward()
        optimizer_G.step()

        #输出训练过程
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch + 1, epochs, i, len(train_loader), d_loss.item(), g_loss.item()))
        if i%100 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())


#模型保存
def model_save(generator):
    torch.save(generator.state_dict(), 'cgan_generator.pth')


if __name__ == '__main__':
    train()
