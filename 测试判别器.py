import torch
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale
from PIL import Image
from src.Discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

discriminator = torch.load("model\\discriminator_model.pth").to(device)

image_path = "generated_images.png"
image = Image.open(image_path)

# 应用转换
transform = Compose([Resize((28, 28)), Grayscale(), ToTensor()])

# 分割图像为 8x8 网格
image_width, image_height = image.size
grid_size = 8
cell_width = image_width // grid_size
cell_height = image_height // grid_size

for row in range(grid_size):
    for col in range(grid_size):
        # 提取子图像
        sub_image = image.crop((col * cell_width, row * cell_height, (col + 1) * cell_width, (row + 1) * cell_height))

        # 对子图像应用转换
        sub_image_tensor = transform(sub_image).unsqueeze(0).to(device)

        # 扁平化子图像
        sub_image_tensor = sub_image_tensor.view(-1, 28 * 28)

        output = discriminator(sub_image_tensor)
        probability = torch.sigmoid(output)

        print(f"Discriminator output for image ({row}, {col}):", output.item())
        print(f"Probability for image ({row}, {col}):", probability.item())


