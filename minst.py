import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
batch_size = 64
num_epochs = 10
learning_rate = 0.001
temperature = 3  # 蒸馏温度

# 加载预训练的ResNet-18模型
teacher_model = models.resnet18(pretrained=True)
num_ftrs = teacher_model.fc.in_features
teacher_model.fc = nn.Linear(num_ftrs, 10)  # 将最后一层的输出改为10类
teacher_model.to(device)
teacher_model.train()  # 设置为训练模式

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片尺寸调整为ResNet-18模型的输入尺寸
    transforms.Grayscale(num_output_channels=3),  # 转换为3通道灰度图像
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 使用ImageNet的均值和标准差
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义一个简单的CNN模型作为学生模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

student_model = SimpleCNN()
student_model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
params = list(student_model.parameters()) + list(teacher_model.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

# 训练学生模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        teacher_outputs = teacher_model(images)
        student_outputs = student_model(images)

        # 计算损失
        distillation_loss = nn.KLDivLoss()(F.log_softmax(student_outputs / temperature, dim=1),
                                           F.softmax(teacher_outputs / temperature, dim=1))
        classification_loss = criterion(student_outputs, labels)
        loss = classification_loss + distillation_loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Classification Loss: {:.4f}, Distillation Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, classification_loss.item(), distillation_loss.item()))

# 保存训练好的学生模型
torch.save(student_model.state_dict(), 'student_model_mnist.ckpt')
