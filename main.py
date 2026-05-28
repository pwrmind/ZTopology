import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# ЧЕСТНЫЙ ЛИНЕЙНЫЙ АНАЛОГ (БЕЗ LEGO-ЛЕСЕНКИ)
# ==========================================
class NoLadderTopology(nn.Module):
    def __init__(self, input_dim=784, reduced_dim=196, num_classes=10):
        super(NoLadderTopology, self).__init__()
        
        # ЭТАП 1: Первичное сжатие (то же самое, что и в версии с лесенкой)
        self.input_compressor = nn.Linear(input_dim, reduced_dim)
        self.activation = nn.ReLU()
        
        # ЭТАП 2: Прямое соединение! Вместо 195 LEGO-блоков мы сразу
        # передаем 196 фичей в промежуточный конус
        self.feature_compressor = nn.Linear(reduced_dim, 96)
        
        # Финальный классификатор на 10 цифр
        self.classifier = nn.Linear(96, num_classes)
        
    def forward(self, x):
        # Распрямляем картинку [Батч, 784]
        x = x.view(x.size(0), -1)
        
        # Шаг 1: Первичное сжатие
        x = self.activation(self.input_compressor(x)) # [Батч, 196]
        
        # Шаг 2: Промежуточный конус (напрямую, без лесенки!)
        x = self.activation(self.feature_compressor(x)) # [Батч, 96]
        
        # Шаг 3: Финальный вердикт
        return self.classifier(x)

# ==========================================
# ЗАПУСК ТЕСТА
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000, shuffle=False)

    model = NoLadderTopology(input_dim=784, reduced_dim=196, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print("Запущен тест архитектуры БЕЗ ЛЕСЕНКИ (Прямое соединение слоев)...")
    
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
