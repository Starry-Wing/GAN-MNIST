import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 MNIST 数据集
dataset = dsets.MNIST(root='./data', train=True,
                      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                      download=True)

# 定义数据加载器
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)


# # 将模型移至 GPU
# generator = Generator().to(device)
# discriminator = Discriminator().to(device)

# 加载生成器模型
generator = torch.load('model/generator_model.pth').to(device)

# 加载判别器模型
discriminator = torch.load('model/discriminator_model.pth').to(device)


# 将损失函数移至 GPU
criterion = nn.BCELoss().to(device)


lr = 0.0002
G_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

# 训练 GAN
num_epochs = 20
for epoch in range(num_epochs):
    print("Epoch:", epoch + 1)
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, -1).to(device)

        # 训练判别器
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, 100).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        discriminator.zero_grad()
        d_loss.backward()
        D_optimizer.step()



        # 训练生成器
        z = torch.randn(batch_size, 100).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)

        g_loss = criterion(outputs, real_labels)
        generator.zero_grad()
        g_loss.backward()
        G_optimizer.step()

        # 输出训练损失信息
        print(f'[{epoch + 1}/{num_epochs}] - D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')


# 保存生成器模型
torch.save(generator, 'model/generator_model.pth')

# 保存判别器模型
torch.save(discriminator, 'model/discriminator_model.pth')

# 生成器生成图像
def create_image():
    # 生成随机噪声

    noise = torch.randn(64, 100).to(device)

    # 使用生成器生成图像
    with torch.no_grad():
        generated_images = generator(noise)

    # 将生成的图像从 [-1, 1] 范围转换回 [0, 1] 范围
    generated_images = (generated_images + 1) / 2

    # 保存生成的图像
    vutils.save_image(generated_images.view(64, 1, 28, 28), 'generated_images.png', nrow=8)

create_image()