import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LegoZStep(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(LegoZStep, self).__init__()
        # Всего две простые линейные трансформации вместо кучи гейтов в GRU!
        self.pixel_weight = nn.Linear(in_channels, hidden_size)
        self.memory_weight = nn.Linear(hidden_size, hidden_size)
        
        # Функция активации для нелинейности (как ReLU в обычных сетях)
        self.activation = nn.ReLU()
        
    def forward(self, current_pixel, previous_memory):
        # 1. Пропускаем пиксель через свои веса
        x_part = self.pixel_weight(current_pixel)
        # 2. Пропускаем прошлую память через свои веса
        h_part = self.memory_weight(previous_memory)
        
        # 3. Смешиваем их и активируем. Чистая лесенка шириной 2!
        new_memory = self.activation(x_part + h_part)
        
        return new_memory

# ==========================================
# 1. СЛОЙ ВИНТОВЫХ LEGO-КОЛОДЦЕВ (Z-СЛОЙ)
# ==========================================
class SpiralLegoWell(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super(SpiralLegoWell, self).__init__()
        self.out_channels = out_channels
        # Порядок движения по спирали: от угла к центру
        self.spiral_indices = [8, 7, 6, 3, 0, 1, 2, 5, 4]
        # self.lego_step = nn.GRUCell(input_size=in_channels, hidden_size=out_channels)
        # Ставим ваш собственный чистый блок:
        self.lego_step = LegoZStep(in_channels=in_channels, hidden_size=out_channels)

        
    def forward(self, x):
        B, C, H, W = x.shape
        x_padded = torch.nn.functional.pad(x, (1, 1, 1, 1)) 
        patches = x_padded.unfold(2, 3, 3).unfold(3, 3, 3) 
        patches = patches.permute(0, 2, 3, 4, 5, 1).flatten(3, 4) 
        flat_patches = patches.reshape(-1, 9, C) 
        
        hx = torch.zeros(flat_patches.size(0), self.out_channels, device=x.device)
        
        for idx in self.spiral_indices:
            pixel_val = flat_patches[:, idx, :]
            hx = self.lego_step(pixel_val, hx)  
            
        return hx.view(B, 10, 10, self.out_channels).permute(0, 3, 1, 2)

# ==========================================
# 2. СБОРКА ПОЛНОЙ СЕТИ ДЛЯ MNIST
# ==========================================
class ZTopologyNet(nn.Module):
    def __init__(self):
        super(ZTopologyNet, self).__init__()
        self.z_layer = SpiralLegoWell(in_channels=1, out_channels=16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, 10) 
        )
        
    def forward(self, x):
        x = self.z_layer(x)       
        x = self.classifier(x)    
        return x

# ==========================================
# 3. ПОДГОТОВКА ДАННЫХ И ОБУЧЕНИЕ
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Загружаем тренировочный датасет
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # ЗАГРУЖАЕМ ТЕСТОВЫЙ ДАТАСЕТ (10 000 картинок)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = ZTopologyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 5
    print(f"Запущено полноценное обучение на {EPOCHS} эпох.")
    print(f"Используемое устройство: {device}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
        train_accuracy = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Блок валидации после каждой эпохи
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
                
        test_accuracy = 100 * correct_test / total_test
        
        print(f"--- ЭПОХА {epoch}/{EPOCHS} ---")
        print(f"Ошибка обучения (Loss): {avg_loss:.4f}")
        print(f"Точность на тренировке: {train_accuracy:.2f}%")
        print(f"Точность на ЭКЗАМЕНЕ (Test Accuracy): {test_accuracy:.2f}%\n")

    print("Полноценный цикл обучения Z-топологии завершен!")
