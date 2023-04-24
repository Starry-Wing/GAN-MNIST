import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from 识别手写数字.src.DigitClassifier import DigitClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
num_epochs = 100
batch_size = 100
learning_rate = 0.001

# 加载 MNIST 数据集
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# # 初始化模型、损失函数和优化器
# model = DigitClassifier().to(device)

# 加载分类器模型参数
model = DigitClassifier().to(device)
model.load_state_dict(torch.load('model/digit_classifier_params.pth'))
model.eval()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练分类器
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch: [{epoch + 1}/{num_epochs}], Step: [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

# 测试分类器
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

# 保存分类器模型参数
torch.save(model.state_dict(), 'model/digit_classifier_params.pth')
