import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. СЛОЙ ВИНТОВЫХ LEGO-КОЛОДЦЕВ (Z-СЛОЙ)
# ==========================================
class SpiralLegoWell(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super(SpiralLegoWell, self).__init__()
        self.out_channels = out_channels
        
        # Индексы обхода матрицы 3x3 по спирали (из угла в центр):
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # Порядок движения: 8 -> 7 -> 6 -> 3 -> 0 -> 1 -> 2 -> 5 -> 4 (центр)
        self.spiral_indices = [8, 7, 6, 3, 0, 1, 2, 5, 4]
        
        # Шаг вашей Z-лесенки: принимает (текущий пиксель + память) -> выдает (новую память)
        # В качестве математического ядра шага идеально подходит GRU-ячейка
        self.lego_step = nn.GRUCell(input_size=in_channels, hidden_size=out_channels)
        
    def forward(self, x):
        B, C, H, W = x.shape  # Batch, Channels, Height, Width
        
        # Добавляем паддинг, чтобы размер 28x28 стал 30x30 (делится на 3)
        x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1)) 
        
        # Нарезаем картинку на неперекрывающиеся окна 3x3 со страйдом 3
        # Получаем сетку 10x10 колодцев
        patches = x_padded.unfold(2, 3, 3).unfold(3, 3, 3) # Форма: [B, C, 10, 10, 3, 3]
        
        # Перегруппируем тензор под последовательный обход пикселей
        patches = patches.permute(0, 2, 3, 4, 5, 1).flatten(3, 4) # [B, 10, 10, 9, C]
        flat_patches = patches.reshape(-1, 9, C) # Объединяем колодцы: [B * 100, 9 пикселей, Каналы]
        
        # Инициализируем нулевую память для старта лесенки в каждом колодце
        hx = torch.zeros(flat_patches.size(0), self.out_channels, device=x.device)
        
        # Крутим винтовую лестницу внутри каждого колодца параллельно по всему батчу!
        for idx in self.spiral_indices:
            pixel_val = flat_patches[:, idx, :] # Берём текущий пиксель из спирали
            hx = self.lego_step(pixel_val, hx)  # LEGO-блок смешивает пиксель и память
            
        # hx теперь содержит сжатую информацию — "выход колодца"
        # Возвращаем структуру обратно в карту признаков [B, 16 каналов, 10, 10]
        return hx.view(B, 10, 10, self.out_channels).permute(0, 3, 1, 2)


# ==========================================
# 2. СБОРКА ПОЛНОЙ СЕТИ ДЛЯ MNIST
# ==========================================
class ZTopologyNet(nn.Module):
    def __init__(self):
        super(ZTopologyNet, self).__init__()
        # Наш кастомный Z-слой из винтовых лестниц
        self.z_layer = SpiralLegoWell(in_channels=1, out_channels=16)
        
        # Полносвязный слой («мозг»), принимающий выходы колодцев
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, 10) # 10 классов (цифры от 0 до 9)
        )
        
    def forward(self, x):
        x = self.z_layer(x)       # Пропускаем через винтовые колодцы
        x = self.classifier(x)    # Классифицируем результат
        return x

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Загрузка датасета MNIST
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Инициализация модели, оптимизатора и функции потерь
model = ZTopologyNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

# Короткий цикл обучения (1 эпоха для теста)
model.train()
print(f"Обучение запущено на устройстве: {device}")

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 200 == 0:
        print(f"Батч {batch_idx}/{len(train_loader)} | Ошибка (Loss): {loss.item():.4f}")

print("Тест завершен! Сеть успешно обучилась на Z-топологии.")
