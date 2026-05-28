import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. ЧЕСТНЫЙ МИКРО-БЛОК LEGO (ШИРИНА 2)
# ==========================================
class PureLegoBlock(nn.Module):
    def __init__(self):
        super(PureLegoBlock, self).__init__()
        # Принимает строго 2 числа: (текущий пиксель, 1 число памяти)
        # Выдает строго 1 число (новую память)
        self.cell = nn.Linear(2, 1)
        self.activation = nn.ReLU()
        
    def forward(self, current_pixel, previous_memory):
        # Объединяем пиксель и память в вектор размера 2
        combined = torch.cat([current_pixel, previous_memory], dim=1)
        return self.activation(self.cell(combined))

# ==========================================
# 2. ЗЕРКАЛЬНАЯ ЛЕСЕНКА Z-ТОПОЛОГИИ
# ==========================================
class HonestZTopology(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(HonestZTopology, self).__init__()
        self.input_dim = input_dim
        
        # Создаем цепочку из N - 1 независимых LEGO блоков
        # Мы упаковываем их в ModuleList, чтобы PyTorch видел их веса
        self.lego_blocks = nn.ModuleList([PureLegoBlock() for _ in range(input_dim - 1)])
        
        # Финальный слой принимает выходы от всех блоков (их ровно N-1)
        # и переводит их в вероятности 10 цифр
        self.classifier = nn.Linear(input_dim - 1, num_classes)
        
    def forward(self, x):
        # Распрямляем картинку в одномерный вектор [Батч, 784]
        x = x.view(x.size(0), -1)
        B = x.size(0)
        
        # Хранилище для выходов каждого блока, которые пойдут на финальный слой
        block_outputs = []
        
        # Инициализируем стартовую пустую память (0.0) для самого первого блока с конца
        memory = torch.zeros(B, 1, device=x.device)
        
        # Движемся строго по вашей схеме: из конца в начало!
        # От пикселя 783 назад к пикселю 1
        for i in range(self.input_dim - 2, -1, -1):
            current_pixel = x[:, i+1].unsqueeze(1) # Берем текущий пиксель
            
            # Пропускаем через i-й блок лесенки
            memory = self.lego_blocks[i](current_pixel, memory)
            
            # Сохраняем шаг лесенки для финального классификатора
            block_outputs.append(memory)
            
        # Собираем все выходы лесенки вместе [Батч, 783]
        # Переворачиваем список, чтобы сохранить порядок от начала к концу
        ladder_features = torch.cat(block_outputs[::-1], dim=1)
        
        # Финальное решение принимает весь каскад лесенки целиком
        return self.classifier(ladder_features)

# ==========================================
# 3. ТЕСТ СЕТИ НА MNIST
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000, shuffle=False)

    model = HonestZTopology(input_dim=784, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    print("Запущен честный экзамен Z-топологии (783 последовательных LEGO-блока)...")
    
    for epoch in range(1, 3): # Погоняем 2 эпохи для честной проверки
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # Валидация
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum().item()
                
        print(f"Эпоха {epoch} | Точность на ЭКЗАМЕНЕ: {correct / 100.0}%")
