import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

data_dir = 'C:\\Users\\TEMP.LAPTOP-EM0D1PRH\\Desktop\\Магистратура\\Искусственный интеллект'

# Загрузка тренировочного набора данных
train_dataset = torchvision.datasets.MNIST(
    root=data_dir, 
    train=True, 
    transform=transforms.ToTensor(),  # Преобразование изображения в тензор
    download=True  # Автоматическая загрузка, если датасет еще не скачан
)

# Загрузка тестового набора данных
test_dataset = torchvision.datasets.MNIST(
    root=data_dir, 
    train=False, 
    transform=transforms.ToTensor(), 
    download=True
)

# Загружаем и нормализуем данные
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование изображения в тензор
    transforms.Normalize((0.1307,), (0.3081,))  # Стандартизация данных
])

# Загрузка тренировочных и тестовых данных
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Создание загрузчиков данных
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # Первый скрытый слой
        self.fc2 = nn.Linear(512, 128)  # Второй скрытый слой
        self.fc3 = nn.Linear(128, 10)   # Выходной слой (10 классов для цифр от 0 до 9)

    def forward(self, x):
        x = x.view(-1, 784)             # Преобразование изображения в вектор
        x = F.relu(self.fc1(x))          # Первая полносвязанная операция с функцией активации ReLU
        x = F.relu(self.fc2(x))          # Вторая полносвязанная операция с функцией активации ReLU
        x = self.fc3(x)                  # Линейный выходной слой
        return F.log_softmax(x, dim=1)   # Логарифмическая функция softmax для получения вероятностей классов

# Создание экземпляра модели
net = Net()

criterion = nn.CrossEntropyLoss()  # Функция потерь для многоклассовой классификации
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # Оптимизатор SGD

epochs = 10  # Количество эпох обучения

# Хранение значений потерь и точности для графика
training_losses = []
testing_accuracies = []

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # Извлекаем входные данные и метки
        
        optimizer.zero_grad()  # Сбрасываем накопленные градиенты
        
        outputs = net(inputs)  # Прямой проход через модель
        loss = criterion(outputs, labels)  # Вычисление потерь
        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Обновление весов модели
        
        # Вывод статистики каждые 200 мини-пакетов
        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200}')
            training_losses.append(running_loss / 200)
            running_loss = 0.0
    
    # Оценка модели на тестовом наборе после каждой эпохи
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # Получение индексов максимального значения
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    testing_accuracy = 100 * correct_test / total_test
    testing_accuracies.append(testing_accuracy)
    print(f'Точность на тестовом наборе после эпохи {epoch+1}: {testing_accuracy:.2f}%')

# Визуализируем рукописную цифру
def show_image(img):
    img = img.reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.show()

# Показываем случайную цифру из тренировочного набора
random_index = torch.randint(len(trainset), (1,))[0]
show_image(trainset[random_index][0].numpy())

# Построение графика изменения точности на тестовом наборе
plt.figure(figsize=(12,6))
plt.plot(range(epochs), testing_accuracies, label="Testing Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Построение графика изменения потерь на тренировочном наборе
plt.figure(figsize=(12,6))
plt.plot(range(epochs), training_losses, label="Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()