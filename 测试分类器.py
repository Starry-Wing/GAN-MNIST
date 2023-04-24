import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor, Resize, Normalize, ToPILImage
import numpy as np
from src.DigitClassifier import DigitClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建分类器模型实例
classifier = DigitClassifier().to(device)
# 加载模型参数
classifier.load_state_dict(torch.load('model\\digit_classifier_params.pth'))
classifier.eval()

# 加载图像
image = Image.open('generated_images.png')

# 预处理图像
transform = transforms.Compose([
    Grayscale(),  # 将图像转换为灰度图像
    Resize((28, 28)),
    ToTensor(),
    Normalize((0.5,), (0.5,)),
])

# 将图像分割为 64 个 28x28 的小图像
images = []
for i in range(8):
    for j in range(8):
        cropped_image = image.crop((j * 28, i * 28, (j + 1) * 28, (i + 1) * 28))
        images.append(transform(cropped_image).unsqueeze(0).to(device))

# 对每个图像进行预测
predictions = []
for img in images:
    with torch.no_grad():
        output = classifier(img)
        pred = output.argmax(dim=1).item()
        predictions.append(pred)

# 以 8x8 的格式打印预测结果
for i in range(8):
    row = predictions[i * 8:(i + 1) * 8]
    print(" ".join([str(n) for n in row]))