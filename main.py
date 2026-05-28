import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. ПРОКАЧАННЫЙ LEGO-БЛОК (С ДВУМЯ СЛОЯМИ)
# ==========================================
class AdvancedLegoBlock(nn.Module):
    def __init__(self, hidden_dim=8):
        super(AdvancedLegoBlock, self).__init__()
        
        # Входной слой блока: расширяет 2 входа (пиксель + память) до hidden_dim
        self.input_layer = nn.Linear(2, hidden_dim)
        self.activation = nn.ReLU()
        # Выходной слой блока: сжимает hidden_dim обратно в 1 число памяти
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, current_pixel, previous_memory):
        # Объединяем два входа в один вектор [Батч, 2]
        combined = torch.cat([current_pixel, previous_memory], dim=1)
        
        # Прогоняем через внутреннюю структуру кубика LEGO
        x = self.input_layer(combined)
        x = self.activation(x)
        new_memory = self.output_layer(x)
        
        # Возвращаем строго 1 число памяти (активируем его, чтобы не было взрыва чисел)
        return torch.tanh(new_memory) 

# ==========================================
# 2. ЗЕРКАЛЬНАЯ ЛЕСЕНКА Z-ТОПОЛОГИИ
# ==========================================
class HonestZTopology(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(HonestZTopology, self).__init__()
        self.input_dim = input_dim
        
        # Создаем цепочку из N - 1 прокачанных LEGO блоков
        self.lego_blocks = nn.ModuleList([AdvancedLegoBlock(hidden_dim=8) for _ in range(input_dim - 1)])
        
        # Финальный классификатор
        self.classifier = nn.Linear(input_dim - 1, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        B = x.size(0)
        
        block_outputs = []
        
        # У первого блока (в самом конце) ПЕРВЫЙ вход — это пиксель 783,
        # а ВТОРОЙ вход (вместо памяти) — это соседний пиксель 782, как на вашей схеме!
        last_pixel = x[:, 783].unsqueeze(1)
        prev_pixel = x[:, 782].unsqueeze(1)
        
        # Запускаем самый первый блок
        memory = self.lego_blocks[self.input_dim - 2](last_pixel, prev_pixel)
        block_outputs.append(memory)
        
        # Гоним цепочку памяти назад к началу (от пикселя 781 до 0)
        for i in range(self.input_dim - 3, -1, -1):
            current_pixel = x[:, i].unsqueeze(1) # Вход с входного слоя
            
            # Блок принимает текущий пиксель и память от предыдущего блока
            memory = self.lego_blocks[i](current_pixel, memory)
            block_outputs.append(memory)
            
        # Собираем все выходы, выравниваем по порядку и отдаем классификатору
        ladder_features = torch.cat(block_outputs[::-1], dim=1)
        return self.classifier(ladder_features)

# ==========================================
# 3. ЗАПУСК ТЕСТА
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000, shuffle=False)

    model = HonestZTopology(input_dim=784, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print("Запущен тест Z-топологии с двухслойными LEGO-блоками...")
    
    for epoch in range(1, 4): # Погоняем 3 эпохи, чтобы увидеть динамику
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
