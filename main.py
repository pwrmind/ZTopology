import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. ПРОКАЧАННЫЙ LEGO-БЛОК
# ==========================================
class AdvancedLegoBlock(nn.Module):
    def __init__(self, hidden_dim=8):
        super(AdvancedLegoBlock, self).__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, current_pixel, previous_memory):
        combined = torch.cat([current_pixel, previous_memory], dim=1)
        x = self.input_layer(combined)
        x = self.activation(x)
        return torch.tanh(self.output_layer(x))

# ==========================================
# 2. ОПТИМИЗИРОВАННАЯ ЭЛЕГАНТНАЯ Z-ТОПОЛОГИЯ
# ==========================================
class ElegantZTopology(nn.Module):
    def __init__(self, input_dim=784, reduced_dim=196, num_classes=10):
        super(ElegantZTopology, self).__init__()
        self.reduced_dim = reduced_dim
        
        # ЭТАП 1: Сжимаем входной слой в 4 раза (с 784 до 196)
        # Это уберет избыточность и объединит пиксели в локальные группы
        self.input_compressor = nn.Linear(input_dim, reduced_dim)
        self.activation = nn.ReLU()
        
        # ЭТАП 2: Лесенка из N - 1 блоков (теперь их всего 195 вместо 783!)
        self.lego_blocks = nn.ModuleList([AdvancedLegoBlock(hidden_dim=8) for _ in range(reduced_dim - 1)])
        
        # ЭТАП 3: Промежуточное сжатие выходов лесенки в 2 раза (со 195 до 96)
        self.feature_compressor = nn.Linear(reduced_dim - 1, 96)
        
        # Финальный классификатор на 10 цифр
        self.classifier = nn.Linear(96, num_classes)
        
    def forward(self, x):
        # Распрямляем картинку [Батч, 784]
        x = x.view(x.size(0), -1)
        B = x.size(0)
        
        # Первичное сжатие входа
        x_compressed = self.activation(self.input_compressor(x)) # [Батч, 196]
        
        block_outputs = []
        
        # Инициализируем первый блок с конца сжатого вектора
        last_val = x_compressed[:, self.reduced_dim - 1].unsqueeze(1)
        prev_val = x_compressed[:, self.reduced_dim - 2].unsqueeze(1)
        
        memory = self.lego_blocks[self.reduced_dim - 2](last_val, prev_val)
        block_outputs.append(memory)
        
        # Гоним укороченную цепочку памяти назад
        for i in range(self.reduced_dim - 3, -1, -1):
            current_val = x_compressed[:, i].unsqueeze(1)
            memory = self.lego_blocks[i](current_val, memory)
            block_outputs.append(memory)
            
        # Собираем выходы лесенки [Батч, 195]
        ladder_features = torch.cat(block_outputs[::-1], dim=1)
        
        # Уменьшаем размер признаков в два раза
        reduced_features = self.activation(self.feature_compressor(ladder_features)) # [Батч, 96]
        
        # Финальный вердикт
        return self.classifier(reduced_features)

# ==========================================
# 3. ЗАПУСК ТЕСТА
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000, shuffle=False)

    model = ElegantZTopology(input_dim=784, reduced_dim=196, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print("Запущен оптимизированный тест Z-топологии (195 блоков + конусное сжатие)...")
    
    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Валидация
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum().item()
                
        print(f"Эпоха {epoch} | Loss: {running_loss/len(train_loader):.4f} | Точность на ЭКЗАМЕНЕ: {correct / 100.0}%")
