import torch
import torchvision.utils as vutils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载生成器模型
generator = torch.load('model/generator_model.pth').to(device)

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