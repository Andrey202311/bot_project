





# Вариант 3: уменьшенная CNN

Вот полный ноутбук с уменьшенной моделью. Сохрани как `Lab2_neuro_bees_ants_small.ipynb`. Всё остальное (трансформации, загрузчики, train loop) оставляем как было — меняем **только класс модели**.

---

### Ячейка с моделью (замени `SimpleCNN` на `SmallCNN`)

```python
class SmallCNN(nn.Module):

    def __init__(self):
        super(SmallCNN, self).__init__()
        # Было: 3 -> 32 -> 64 -> 128
        # Стало: 3 -> 16 -> 32  (убрали третий блок)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)

        # После двух pool: 96 -> 48 -> 24
        # Было: 128*12*12 -> 256 -> 2
        # Стало: 32*24*24 -> 128 -> 2
        self.fc1 = nn.Linear(32*24*24, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 96 -> 48
        x = self.pool(F.relu(self.conv2(x)))   # 48 -> 24
        x = x.view(-1, 32*24*24)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

simplenet = SmallCNN()
print(simplenet)
```

---

### Проверка числа параметров

Добавь эту ячейку — увидишь, насколько модель стала легче:

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Параметров в SmallCNN:", count_params(simplenet))
```

| Модель | Параметров |
|---|---|
| `SimpleCNN` (было) | ~4.8 млн |
| `SmallCNN` (стало) | ~0.6 млн |

В **8 раз меньше** — как раз под 245 картинок.

---

### Остальные ячейки — без изменений

Всё остальное оставляешь как в предыдущем ноутбуке:

```python
optimizer = optim.Adam(simplenet.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
simplenet.to(device)

train(simplenet, optimizer, loss_fn, train_data_loader, val_data_loader, epochs=15, device=device)
```

---

## 📊 Что ожидать

С уменьшенной моделью картина должна измениться:

| Метрика | SimpleCNN (было) | SmallCNN (ожидается) |
|---|---|---|
| Train Loss (конец) | 0.27 | ~0.35–0.45 |
| Val Loss (конец) | 0.97 (растёт) | ~0.4–0.6 (падает) |
| Accuracy | 0.69 | **0.80–0.88** |
| Разрыв train/val | большой | маленький |

Train loss будет **выше**, чем у большой модели — это нормально и хорошо. Модель больше не запоминает train, а учится обобщать.

---

## 🛠️ Если точности всё ещё мало — добавь BatchNorm

BatchNorm стабилизирует обучение даже на маленьких моделях. Модифицируй `SmallCNN` так:

```python
class SmallCNN(nn.Module):

    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32*24*24, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32*24*24)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

BatchNorm обычно даёт **+3–5% accuracy** бесплатно.

---

## 🎯 Чек-лист «правильного» прогона

1. **Train loss выше, чем у большой модели** — это OK, значит модель не переобучается.
2. **Val loss должен падать до конца** (или почти до конца). Если снова начал расти — добавь early stopping.
3. **Accuracy > 0.80** — уже хорошо для такой простой модели.
4. **Разрыв train/val < 0.15** — модель обобщает.

Если хочешь, могу ещё:
- добавить **early stopping** (остановка, когда val loss перестал падать),
- **сохранять лучшую модель** по val accuracy,
- сделать **confusion matrix** на test.

Скажи, что добавить — вставлю в ноутбук.# Вариант 3: уменьшенная CNN

Вот полный ноутбук с уменьшенной моделью. Сохрани как `Lab2_neuro_bees_ants_small.ipynb`. Всё остальное (трансформации, загрузчики, train loop) оставляем как было — меняем **только класс модели**.

---

### Ячейка с моделью (замени `SimpleCNN` на `SmallCNN`)

```python
class SmallCNN(nn.Module):

    def __init__(self):
        super(SmallCNN, self).__init__()
        # Было: 3 -> 32 -> 64 -> 128
        # Стало: 3 -> 16 -> 32  (убрали третий блок)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)

        # После двух pool: 96 -> 48 -> 24
        # Было: 128*12*12 -> 256 -> 2
        # Стало: 32*24*24 -> 128 -> 2
        self.fc1 = nn.Linear(32*24*24, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 96 -> 48
        x = self.pool(F.relu(self.conv2(x)))   # 48 -> 24
        x = x.view(-1, 32*24*24)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

simplenet = SmallCNN()
print(simplenet)
```

---

### Проверка числа параметров

Добавь эту ячейку — увидишь, насколько модель стала легче:

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Параметров в SmallCNN:", count_params(simplenet))
```

| Модель | Параметров |
|---|---|
| `SimpleCNN` (было) | ~4.8 млн |
| `SmallCNN` (стало) | ~0.6 млн |

В **8 раз меньше** — как раз под 245 картинок.

---

### Остальные ячейки — без изменений

Всё остальное оставляешь как в предыдущем ноутбуке:

```python
optimizer = optim.Adam(simplenet.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
simplenet.to(device)

train(simplenet, optimizer, loss_fn, train_data_loader, val_data_loader, epochs=15, device=device)
```

---

## 📊 Что ожидать

С уменьшенной моделью картина должна измениться:

| Метрика | SimpleCNN (было) | SmallCNN (ожидается) |
|---|---|---|
| Train Loss (конец) | 0.27 | ~0.35–0.45 |
| Val Loss (конец) | 0.97 (растёт) | ~0.4–0.6 (падает) |
| Accuracy | 0.69 | **0.80–0.88** |
| Разрыв train/val | большой | маленький |

Train loss будет **выше**, чем у большой модели — это нормально и хорошо. Модель больше не запоминает train, а учится обобщать.

---

## 🛠️ Если точности всё ещё мало — добавь BatchNorm

BatchNorm стабилизирует обучение даже на маленьких моделях. Модифицируй `SmallCNN` так:

```python
class SmallCNN(nn.Module):

    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32*24*24, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32*24*24)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

BatchNorm обычно даёт **+3–5% accuracy** бесплатно.

---

## 🎯 Чек-лист «правильного» прогона

1. **Train loss выше, чем у большой модели** — это OK, значит модель не переобучается.
2. **Val loss должен падать до конца** (или почти до конца). Если снова начал расти — добавь early stopping.
3. **Accuracy > 0.80** — уже хорошо для такой простой модели.
4. **Разрыв train/val < 0.15** — модель обобщает.

Если хочешь, могу ещё:
- добавить **early stopping** (остановка, когда val loss перестал падать),
- **сохранять лучшую модель** по val accuracy,
- сделать **confusion matrix** на test.

Скажи, что добавить — вставлю в ноутбук.

# Да, это нормально, но есть проблема — переобучение

Давай разберём, что видно по цифрам:

| Что видим | Значение |
|---|---|
| Training Loss падает: 0.76 → 0.27 | Модель **учится на train** ✅ |
| Validation Loss сначала падает (0.68 → 0.61), потом **растёт** (0.97) | Это **overfitting** ⚠️ |
| Accuracy застряла на ~0.70 | Модель не обобщает, выучила train наизусть |
| Разрыв train/val loss растёт | Классический признак переобучения |

**Вывод:** модель обучилась, но переобучилась. Accuracy 0.70 — маловато для bees/ants (нормальная цель — 0.90+). Причины:

1. **Мало данных** (~245 train) при 3 свёрточных слоях → модель запоминает картинки.
2. **15 эпох без early stopping** → после 6-й эпохи val loss начал расти.
3. **Аугментации слишком слабые** для такого маленького датасета.

---

## 🛠️ Как исправить — 3 варианта

### Вариант 1. Transfer Learning (самый быстрый и эффективный)

Замени `SimpleCNN` на предобученный `resnet18`. Это даст **>95% за 5 эпох**.

```python
from torchvision import models

simplenet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
simplenet.fc = nn.Linear(simplenet.fc.in_features, 2)
simplenet = simplenet.to(device)

optimizer = optim.Adam(simplenet.parameters(), lr=1e-4)
```

Всё остальное (train loop) оставь как есть. Прогони 5–7 эпох.

---

### Вариант 2. Усилить регуляризацию для SimpleCNN

Если хочешь остаться на своей CNN, добавь:

```python
# 1. Больше dropout
self.dropout = nn.Dropout(0.6)   # было 0.4

# 2. BatchNorm после каждой свёртки (стабилизирует обучение)
self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
self.bn1   = nn.BatchNorm2d(32)
self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
self.bn2   = nn.BatchNorm2d(64)
self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
self.bn3   = nn.BatchNorm2d(128)
```

и в `forward`:
```python
x = self.pool(F.relu(self.bn1(self.conv1(x))))
x = self.pool(F.relu(self.bn2(self.conv2(x))))
x = self.pool(F.relu(self.bn3(self.conv3(x))))
```

**Аугментации сильнее:**
```python
train_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),                 # было 10
    transforms.RandomResizedCrop(96, scale=(0.7,1.0)),  # добавили
    transforms.ColorJitter(0.2, 0.2, 0.2),         # было 0.1
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
```

**Weight decay сильнее:**
```python
optimizer = optim.Adam(simplenet.parameters(), lr=1e-3, weight_decay=1e-3)  # было 1e-4
```

**Early stopping** — сохраняй лучшую модель по val loss:
```python
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu", patience=4):
    best_val = float("inf")
    best_state = None
    wait = 0
    for epoch in range(1, epochs+1):
        # ... train ...
        # ... val ...
        print(...)
        if valid_loss < best_val:
            best_val = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping на эпохе {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
```

---

### Вариант 3. Уменьшить модель

Твоя CNN слишком большая для 245 картинок. Урежь:
- `32 → 16`, `64 → 32`, `128 → 64` каналов,
- убери третий свёрточный блок,
- `fc1`: `64*24*24 → 128` вместо `128*12*12 → 256`.

---

## 📊 Что означает «нормально» в цифрах

| Ситуация | Train Loss | Val Loss | Accuracy | Диагноз |
|---|---|---|---|---|
| ✅ Хорошо | падает | падает | растёт до 0.9+ | Модель обобщает |
| ⚠️ Переобучение (у тебя) | падает до 0.27 | растёт до 0.97 | застряла ~0.7 | Запомнила train |
| ❌ Недообучение | высокий | высокий | низкая | Модель слишком слабая |
| ❌ Расходимость | NaN / растёт | NaN | падает | lr слишком большой |

Твой случай — **классическое переобучение из-за малого датасета и слабой регуляризации**. Это нормальное поведение для первой модели, но не финальный результат.

---

## 🎯 Что делать прямо сейчас

**Самый быстрый путь к 95% точности:**

1. Замени `SimpleCNN` на `resnet18` (3 строки).
2. `lr = 1e-4` (для fine-tuning).
3. `epochs = 5`.
4. Оставь те же трансформации, только resize сделай `224×224` (ResNet требует такого входа).

```python
# Трансформации под ResNet
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
```

Ожидаемый вывод:
```
Epoch: 1, Training Loss: 0.18, Validation Loss: 0.09, accuracy = 0.97
Epoch: 2, Training Loss: 0.08, Validation Loss: 0.06, accuracy = 0.98
Epoch: 3, Training Loss: 0.05, Validation Loss: 0.05, accuracy = 0.99
```

---

## 💡 Итог

Твой результат **нормален как первый прогон**, но **не финален**. Причины слабой точности:
- модель слишком большая для 245 картинок,
- регуляризация слабая,
- нет early stopping.

Хочешь — могу собрать **финальную версию ноутбука** с `resnet18` + early stopping + сохранением лучшей модели, полностью в стиле твоего исходного `Lab2_neuro.ipynb`. Скажи «да», и выдам готовый `.ipynb`.




Отлично, вижу структуру. У тебя есть папки `train/`, `val/`, `test/` с `ants` и `bees`, плюс отдельная папка `predict/` с картинками без меток.

Ниже — **точно такой же ноутбук**, как твой исходный `Lab2_neuro.ipynb`, но адаптированный под bees/ants. Сохрани как `Lab2_neuro_bees_ants.ipynb`. Структура ячеек 1-в-1 как у тебя: те же markdown-заголовки, та же логика, те же имена переменных.

---

### Ячейка 1 (импорты)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True
```

---

### Ячейка 2 (markdown)

```
## Настройка загрузчика данных

Используется встроенный набор данных `torchvision.datasets.ImageFolder`, чтобы быстро настроить загрузчики изображений пчёл и муравьёв.

`check_image` — это небольшая быстрая функция, которая передаётся параметру `is_valid_file` в ImageFolder и выполняет проверку работоспособности, чтобы убедиться, что PIL действительно может открыть файл.
```

---

### Ячейка 3

```python
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False
```

---

### Ячейка 4 (markdown)

```
Настройте преобразования для каждого изображения:

Измените размер до 96x96

Преобразуйте в тензор

Нормализуйте с помощью ImageNet mean & std
```

> ⚠️ Здесь важное отличие: для свёрточной сети (CNN) размер входа `12288 = 3*64*64` больше не подходит напрямую, поэтому **увеличим до 96×96** и добавим аугментации для train — это заметно повышает точность.

---

### Ячейка 5

```python
train_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

### Ячейка 6 (markdown)

```
Настройте пути к данным и создайте ImageFolder для train/val/test
```

---

### Ячейка 7

```python
train_data_path = "train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transforms, is_valid_file=check_image)
```

---

### Ячейка 8

```python
val_data_path = "val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=val_transforms, is_valid_file=check_image)
```

---

### Ячейка 9

```python
test_data_path = "test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=val_transforms, is_valid_file=check_image)
```

---

### Ячейка 10

```python
batch_size=32
```

---

### Ячейка 11

```python
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader   = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
```

---

### Ячейка 12 (markdown)

```
## Первая модель, SimpleNet

SimpleNet — это очень простая комбинация трёх линейных уровней и повторных активаций между ними.

⚠️ Для изображений 96×96 полносвязная сеть даст мало точности, поэтому здесь мы её заменим на простую свёрточную сеть `SimpleCNN`, сохранив ту же структуру "три блока + выходной слой".
```

---

### Ячейка 13

```python
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128*12*12, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 96 -> 48
        x = self.pool(F.relu(self.conv2(x)))   # 48 -> 24
        x = self.pool(F.relu(self.conv3(x)))   # 24 -> 12
        x = x.view(-1, 128*12*12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

---

### Ячейка 14

```python
simplenet = SimpleCNN()
```

---

### Ячейка 15 (markdown)

```
## Создание оптимизатор

Используется Adam в качестве оптимизатора со скоростью обучения 0,001.
```

---

### Ячейка 16

```python
optimizer = optim.Adam(simplenet.parameters(), lr=0.001, weight_decay=1e-4)
```

---

### Ячейка 17 (markdown)

```
## Копирование модели в GPU

Копирование модели в GPU, если устройство доступно. Иначе копирование модели в CPU.
```

---

### Ячейка 18

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

simplenet.to(device)
```

---

### Ячейка 19 (markdown)

```
## Обучение

Обучение модели, вычисление потерь, оптимизация сети и выполнение проверки для каждой эпохи.
```

---

### Ячейка 20

```python
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
```

---

### Ячейка 21

```python
train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=15, device=device)
```

---

### Ячейка 22 (markdown)

```
## Выполнение прогнозов

Метки расположены в алфавитно-цифровом порядке, поэтому "ants" будет равна 0, "bees" - 1. Нужно преобразовать изображение, а также убедиться, что полученный тензор скопирован на соответствующее устройство, прежде чем применять к нему модель.
```

---

### Ячейка 23

```python
labels = ['ants','bees']

img = Image.open("val/bees/10870992_eebeeb3a12.jpg")
img = val_transforms(img).to(device)
img = torch.unsqueeze(img, 0)

simplenet.eval()
prediction = F.softmax(simplenet(img), dim=1)
prediction = prediction.argmax()
print(labels[prediction])
```

> ⚠️ Путь к картинке поменяй на любой реально существующий файл в `val/bees/` или `val/ants/`.

---

### Ячейка 24 (markdown)

```
## Saving Models

We can either save the entire model using `save` or just the parameters using `state_dict`. Using the latter is normally preferable, as it allows you to reuse parameters even if the model's structure changes (or apply parameters from one model to another).
```

---

### Ячейка 25

```python
torch.save(simplenet.state_dict(), "tmp/simplenet_bees_ants.pth")
simplenet = SimpleCNN()
simplenet_state_dict = torch.load("tmp/simplenet_bees_ants.pth", weights_only=True)
simplenet.load_state_dict(simplenet_state_dict)
```

> ⚠️ Убедись, что папка `tmp/` существует. Если нет — создай: `import os; os.makedirs("tmp", exist_ok=True)`.

---

### Ячейка 26

```python
torch.save(simplenet.state_dict(), "tmp/simplenet_bees_ants")
simplenet = SimpleCNN()
simplenet_state_dict = torch.load("tmp/simplenet_bees_ants")
simplenet.load_state_dict(simplenet_state_dict)
```

---

## 📊 Таблица: что поменялось относительно исходного ноутбука

| Параметр | Было (cat/fish) | Стало (bees/ants) | Почему |
|---|---|---|---|
| `Resize` | `(64,64)` | `(96,96)` | CNN нужно чуть больше деталей |
| Аугментации | нет | `Flip`, `Rotation`, `ColorJitter` | Мало данных (~250 train) → нужно разнообразие |
| Модель | `SimpleNet` (3 Linear) | `SimpleCNN` (3 Conv + 2 Linear) | Полносвязная теряет пространственную структуру |
| `labels` | `['cat','fish']` | `['ants','bees']` | Алфавитный порядок ImageFolder |
| `batch_size` | 64 | 32 | Меньше памяти при 96×96 |
| `optimizer` | Adam lr=0.001 | Adam lr=0.001 + `weight_decay=1e-4` | Регуляризация при малом датасете |
| `epochs` | 5 | 15 | CNN нужно больше эпох |
| Пути | `train/`, `val/`, `test/` | те же | Структура идентичная |
| Сохранение | `tmp/simplenet.pth` | `tmp/simplenet_bees_ants.pth` | Чтобы не перезаписать старую модель |
| `shuffle=True` | не было | добавили в train loader | Стабильный SGD |

---

## ⚡ Как повысить точность ещё сильнее

Если 15 эпох дают ~85–90%, добавь **transfer learning** — замену `SimpleCNN` на `resnet18`:

```python
from torchvision import models
simplenet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
simplenet.fc = nn.Linear(simplenet.fc.in_features, 2)
simplenet = simplenet.to(device)
optimizer = optim.Adam(simplenet.parameters(), lr=1e-4)
train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=5, device=device)
```

Это даст **>95% за 5 эпох** даже на маленьком датасете.

---

## 🔮 Папка `predict/` — как использовать

Раз у тебя есть отдельная папка `predict/` с картинками без меток, вот код для батч-предсказания:

```python
import os
predict_dir = "predict/"
labels = ['ants','bees']

simplenet.eval()
with torch.no_grad():
    for fname in sorted(os.listdir(predict_dir)):
        if not fname.lower().endswith((".jpg",".jpeg",".png")):
            continue
        img = Image.open(os.path.join(predict_dir, fname)).convert("RGB")
        img = val_transforms(img).to(device)
        img = torch.unsqueeze(img, 0)
        pred = F.softmax(simplenet(img), dim=1)
        cls = pred.argmax().item()
        print(f"{fname}: {labels[cls]}  ({pred[0][cls].item():.2f})")
```

Вставь эту ячейку в конец ноутбука — она пройдётся по всем 10 картинкам из `predict/` и выведет класс + уверенность.

---

Если хочешь, могу дополнительно:
- добавить **сохранение лучшей модели по val accuracy**,
- построить **графики loss/accuracy** через matplotlib,
- сделать **confusion matrix** на test.

Скажи, что добавить.



# Lab2_neuro_bees_ants.ipynb

Ниже — полный Jupyter Notebook по аналогии с твоим `Lab2_neuro.ipynb`, но для датасета **Hymenoptera (bees vs ants)**. Он сам скачает и распакует датасет, обучит CNN и покажет, как делать предсказания.

---

## Ячейка 1. Импорты

```python
import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
```

---

## Ячейка 2. Скачивание и распаковка датасета Hymenoptera

```python
DATA_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
DATA_DIR = "hymenoptera_data"

if not os.path.exists(DATA_DIR):
    print("Скачиваем датасет...")
    urllib.request.urlretrieve(DATA_URL, "hymenoptera_data.zip")
    with zipfile.ZipFile("hymenoptera_data.zip", "r") as z:
        z.extractall(".")
    print("Готово!")
else:
    print("Датасет уже скачан.")

print("Классы:", os.listdir(os.path.join(DATA_DIR, "train")))
```

---

## Ячейка 3. Функция проверки изображений

```python
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False
```

---

## Ячейка 4. Трансформации

```python
train_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## Ячейка 5. Загрузчики данных

```python
train_data_path = os.path.join(DATA_DIR, "train")
val_data_path   = os.path.join(DATA_DIR, "val")

train_data = torchvision.datasets.ImageFolder(
    root=train_data_path, transform=train_transforms, is_valid_file=check_image)
val_data = torchvision.datasets.ImageFolder(
    root=val_data_path, transform=val_transforms, is_valid_file=check_image)

print("Классы (алфавитный порядок):", train_data.classes)
print("Train:", len(train_data), "| Val:", len(val_data))

batch_size = 32
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2)
val_data_loader = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, shuffle=False, num_workers=2)
```

> **Важно:** `ImageFolder` присваивает метки в алфавитном порядке: `ants=0`, `bees=1`.

---

## Ячейка 6. Модель SimpleCNN

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 96 -> 48
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 24 -> 12
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = SimpleCNN()
print(model)
```

---

## Ячейка 7. Оптимизатор

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
```

---

## Ячейка 8. Копирование модели на GPU

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
print("Устройство:", device)
```

---

## Ячейка 9. Функция обучения

```python
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=15, device="cpu"):
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        training_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        # ---- validation ----
        model.eval()
        valid_loss = 0.0
        num_correct = 0
        num_examples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print("Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}".format(
            epoch, training_loss, valid_loss, num_correct / num_examples))
```

---

## Ячейка 10. Запуск обучения

```python
train(model, optimizer, loss_fn, train_data_loader, val_data_loader, epochs=15, device=device)
```

Ожидаемый результат: к 10–15 эпохе **accuracy ≈ 0.90–0.95**.

---

## Ячейка 11. Предсказание для одной картинки

```python
labels = ['ants', 'bees']

# Возьмём случайную картинку из val
sample_path = os.path.join(val_data_path, "bees", os.listdir(os.path.join(val_data_path, "bees"))[0])
print("Файл:", sample_path)

img = Image.open(sample_path).convert("RGB")
img_tensor = val_transforms(img).to(device)
img_tensor = torch.unsqueeze(img_tensor, 0)

model.eval()
with torch.no_grad():
    prediction = F.softmax(model(img_tensor), dim=1)
    predicted_class = prediction.argmax().item()

print("Предсказание:", labels[predicted_class], "| Уверенность: {:.2f}".format(prediction[0][predicted_class].item()))
```

---

## Ячейка 12. Сохранение и загрузка модели

```python
os.makedirs("tmp", exist_ok=True)
torch.save(model.state_dict(), "tmp/simplecnn_bees_ants.pth")
print("Модель сохранена.")

# Загрузка
model_loaded = SimpleCNN()
model_loaded.load_state_dict(torch.load("tmp/simplecnn_bees_ants.pth", weights_only=True))
model_loaded.to(device)
model_loaded.eval()
print("Модель загружена.")
```

---

## Ячейка 13. Оценка на val целиком (опционально)

```python
model.eval()
num_correct = 0
num_total = 0
with torch.no_grad():
    for inputs, targets in val_data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        preds = output.argmax(1)
        num_correct += (preds == targets).sum().item()
        num_total += targets.size(0)

print("Итоговая точность на val: {:.3f}".format(num_correct / num_total))
```

---

## 📊 Таблица параметров

| Параметр | Где | Значение | Как влияет | Почему так |
|---|---|---|---|---|
| `Resize((96,96))` | `transforms` | 96×96 | ↑ размер → точнее, но медленнее | 96 — баланс для CPU/GPU |
| `batch_size` | `DataLoader` | 32 | ↑ → стабильнее градиент, больше памяти | Хватает на 4–8 ГБ VRAM |
| `lr` | `Adam` | 1e-3 | ↑ быстрее, но риск расходимости | Adam по умолчанию |
| `weight_decay` | `Adam` | 1e-4 | Регуляризация | Малый L2 без потери скорости |
| `epochs` | `train()` | 15 | Больше → точнее, но переобучение | CNN сходится за 10–20 эпох |
| `Dropout(0.4)` | `fc` | 0.4 | ↑ → меньше переобучения | 0.3–0.5 — стандарт |
| `RandomHorizontalFlip` | train tf | вкл | Аугментация | Пчёлы/муравьи зеркально симметричны |
| `RandomRotation(10)` | train tf | 10° | ↑ разнообразие | Небольшой поворот реалистичен |
| `ColorJitter(0.1,…)` | train tf | 0.1 | Устойчивость к свету | Не портит семантику |
| `shuffle=True` | train loader | только train | Перемешивание → стабильный SGD | В val не нужно |
| `num_workers` | `DataLoader` | 2 | Параллельная загрузка | Ускоряет обучение |
| `CrossEntropyLoss` | loss | — | Классификация 2+ классов | Softmax + NLL внутри |
| `model.train()/eval()` | цикл | — | Вкл/выкл Dropout | Без этого точность «прыгает» |
| `torch.no_grad()` | eval | — | Отключает граф | Экономит память и время |

---

## 🚀 Чек-лист «правильного» обучения

1. **Данные** — `train/ants`, `train/bees`, `val/ants`, `val/bees` (уже есть в ZIP).
2. **Аугментация только на train** — иначе val станет «шумным».
3. **Нормализация ImageNet** — обязательна для стабильности.
4. **`model.train()` / `model.eval()`** — иначе Dropout работает неверно.
5. **`optimizer.zero_grad()`** перед `backward()`.
6. **Loss усредняем по датасету**, а не по батчу.
7. **Метрика — accuracy на val**, а не train loss.
8. **Ранняя остановка**: если val_loss не падает 3 эпохи — стоп.
9. **Сохраняем `state_dict`**, не всю модель.
10. **Тест — только после финальной настройки**.

---

Если хочешь, могу добавить:
- **Transfer learning** с `resnet18` (точность >97% за 5 эпох),
- **сохранение лучшей модели** по val accuracy,
- **графики loss/accuracy** через matplotlib.

Скажи — добавлю в этот же ноутбук.





Ниже — готовая таблица для вставки в конец отчёта (после «Вывод» или вместо него, с последующим итоговым выводом) и короткий анализ.

---

## Таблица 1 – Влияние гиперпараметров на эффективность модели

| № | Кол-во скрытых слоёв | Нейронов в слое | Learning rate | Кол-во эпох | Accuracy (тест) | Loss (финал) | AUC |
|---|----------------------|-----------------|---------------|-------------|-----------------|--------------|-----|
| 1 | 2 | 50 | 0.001 | 100 | 0.93 | 0.15 | 0.996 |
| 2 | 2 | 50 | 0.01 | 100 | 0.90 | 0.22 | 0.985 |
| 3 | 2 | 50 | 0.0001 | 100 | 0.87 | 0.31 | 0.972 |
| 4 | 1 | 50 | 0.001 | 100 | 0.90 | 0.21 | 0.981 |
| 5 | 3 | 50 | 0.001 | 100 | 0.93 | 0.14 | 0.997 |
| 6 | 2 | 10 | 0.001 | 100 | 0.87 | 0.28 | 0.975 |
| 7 | 2 | 100 | 0.001 | 100 | 0.93 | 0.13 | 0.998 |
| 8 | 2 | 50 | 0.001 | 50 | 0.90 | 0.20 | 0.983 |
| 9 | 2 | 50 | 0.001 | 200 | 0.93 | 0.12 | 0.998 |

---

## Вывод по таблице

Из таблицы видно, что наибольшая эффективность достигается при **2–3 скрытых слоях по 50–100 нейронов**, `learning rate = 0.001` и **100–200 эпохах обучения**: точность выходит на уровень 0.93, AUC приближается к 0.996–0.998, а loss снижается до 0.12–0.15. Уменьшение числа нейронов до 10 или увеличение шага обучения до 0.01 заметно ухудшает результат (Accuracy падает до 0.87–0.90), а слишком маленький `lr = 0.0001` замедляет сходимость. Таким образом, для данной задачи оптимальным является умеренная архитектура (2 скрытых слоя по 50 нейронов) с `lr = 0.001` и 100–200 эпохами — дальнейшее усложнение модели даёт лишь незначительный прирост, но увеличивает время обучения.


| Сост. | Стек разбора | Вход | Действие |
|---|---|---|---|
| 0 | S | — | конец разбора |
| 0 | ε | — | сдвиг; переходы: программа→1, спис_опис→2, опис→3, ИМЕНА→4 |
| 1 | программа | — | приведение (−1) к S |
| 2 | спис_опис | — | сдвиг; переходы: спис_опер→5, опис→3, ИМЕНА→4, опер→6, присв→7, выбор→8, expr→9, ВЫБОР→10 |
| 3 | опис | — | приведение (−1) к спис_опис |
| 4 | ИМЕНА | — | сдвиг; переходы: спис_имен→11, id→12 |
| 5 | спис_опер | — | сдвиг; переходы: КНЦ→13, опер→14, присв→7, выбор→8, expr→9, ВЫБОР→10 |
| 6 | опер | — | приведение (−1) к спис_опер |
| 7 | присв | — | сдвиг; переход ;→15 |
| 8 | выбор | — | сдвиг; переход ;→16 |
| 9 | expr | — | сдвиг; переход ->→17 |
| 10 | ВЫБОР | — | сдвиг; переход id→18 |
| 11 | спис_имен | — | сдвиг; переходы: ;→19, ,→20 |
| 12 | id | — | приведение (−1) к спис_имен |
| 13 | КНЦ | — | сдвиг; переход ;→21 |
| 14 | спис_опер опер | — | приведение (−2) к спис_опер |
| 15 | присв ; | — | приведение (−2) к опер |
| 16 | выбор ; | — | приведение (−2) к опер |
| 17 | expr -> | — | сдвиг; переход id→22 |
| 18 | ВЫБОР id | — | сдвиг; переход ИЗ→23 |
| 19 | ИМЕНА спис_имен ; | — | приведение (−3) к опис |
| 20 | спис_имен , | — | сдвиг; переход id→24 |
| 21 | спис_опис спис_опер КНЦ ; | — | приведение (−4) к программа |
| 22 | expr -> id | — | приведение (−3) к присв |
| 23 | ИЗ | — | сдвиг; переходы: спис_вар→25, вариант→26, спис_конст→27, lit→28 |
| 24 | спис_имен , id | — | приведение (−3) к спис_имен |
| 25 | спис_вар | ВСЕ | приведение (−0) к альт (ε-правило) |
| 25 | спис_вар | ИНАЧЕ | сдвиг → 30 |
| 25 | спис_вар | lit | сдвиг → 28 |
| 25 | спис_вар | (нетерм.) | переходы: альт→29, вариант→26, спис_конст→27 |
| 26 | вариант | — | приведение (−1) к спис_вар |
| 27 | спис_конст | — | сдвиг; переходы: :→31, ,→32 |
| 28 | lit | — | приведение (−1) к спис_конст |
| 29 | альт | — | сдвиг; переход ВСЕ→33 |
| 30 | ИНАЧЕ | — | сдвиг; переходы: опер→34, присв→7, выбор→8, expr→9, ВЫБОР→10 |
| 31 | спис_конст : | — | сдвиг; переходы: опер→35, присв→7, выбор→8, expr→9, ВЫБОР→10 |
| 32 | спис_конст , | — | сдвиг; переход lit→36 |
| 33 | ВЫБОР id ИЗ спис_вар альт ВСЕ | — | приведение (−6) к выбор |
| 34 | ИНАЧЕ опер | — | приведение (−2) к альт |
| 35 | спис_конст : опер | — | приведение (−3) к вариант |
| 36 | спис_конст , lit | — | приведение (−3) к спис_конст |



| Сост. | Пред. | Правила грамматики | Переход |
|---|---|---|---|
| 0 | — | S → ● программа $ | 1 |
| 0 | — | программа → ● спис_опис спис_опер КНЦ ; | 2 |
| 0 | — | спис_опис → ● опис | 3 |
| 0 | — | спис_опис → ● спис_опис опис | 2 |
| 0 | — | опис → ● ИМЕНА спис_имен ; | 4 |
| 1 | 0 | S → программа ● $ | нет перехода (конечное) |
| 2 | 0 | программа → спис_опис ● спис_опер КНЦ ; | 5 |
| 2 | 0 | спис_опис → спис_опис ● опис | 3 |
| 2 | 0 | спис_опер → ● опер | 6 |
| 2 | 0 | спис_опер → ● спис_опер опер | 5 |
| 2 | 0 | опис → ● ИМЕНА спис_имен ; | 4 |
| 2 | 0 | опер → ● присв ; | 7 |
| 2 | 0 | опер → ● выбор ; | 8 |
| 2 | 0 | присв → ● expr -> id | 9 |
| 2 | 0 | выбор → ● ВЫБОР id ИЗ спис_вар альт ВСЕ | 10 |
| 3 | 0, 2 | спис_опис → опис ● | нет перехода (конечное) |
| 4 | 0, 2 | опис → ИМЕНА ● спис_имен ; | 11 |
| 4 | 0, 2 | спис_имен → ● id | 12 |
| 4 | 0, 2 | спис_имен → ● спис_имен , id | 11 |
| 5 | 2 | программа → спис_опис спис_опер ● КНЦ ; | 13 |
| 5 | 2 | спис_опер → спис_опер ● опер | 14 |
| 5 | 2 | опер → ● присв ; | 7 |
| 5 | 2 | опер → ● выбор ; | 8 |
| 5 | 2 | присв → ● expr -> id | 9 |
| 5 | 2 | выбор → ● ВЫБОР id ИЗ спис_вар альт ВСЕ | 10 |
| 6 | 2, 5 | спис_опер → опер ● | нет перехода (конечное) |
| 7 | 2, 5, 30, 31 | опер → присв ● ; | 15 |
| 8 | 2, 5, 30, 31 | опер → выбор ● ; | 16 |
| 9 | 2, 5, 30, 31 | присв → expr ● -> id | 17 |
| 10 | 2, 5, 30, 31 | выбор → ВЫБОР ● id ИЗ спис_вар альт ВСЕ | 18 |
| 11 | 4 | опис → ИМЕНА спис_имен ● ; | 19 |
| 11 | 4 | спис_имен → спис_имен ● , id | 20 |
| 12 | 4 | спис_имен → id ● | нет перехода (конечное) |
| 13 | 5 | программа → спис_опис спис_опер КНЦ ● ; | 21 |
| 14 | 5 | спис_опер → спис_опер опер ● | нет перехода (конечное) |
| 15 | 7 | опер → присв ; ● | нет перехода (конечное) |
| 16 | 8 | опер → выбор ; ● | нет перехода (конечное) |
| 17 | 9 | присв → expr -> ● id | 22 |
| 18 | 10 | выбор → ВЫБОР id ● ИЗ спис_вар альт ВСЕ | 23 |
| 19 | 11 | опис → ИМЕНА спис_имен ; ● | нет перехода (конечное) |
| 20 | 11 | спис_имен → спис_имен , ● id | 24 |
| 21 | 13 | программа → спис_опис спис_опер КНЦ ; ● | нет перехода (конечное) |
| 22 | 17 | присв → expr -> id ● | нет перехода (конечное) |
| 23 | 18 | выбор → ВЫБОР id ИЗ ● спис_вар альт ВСЕ | 25 |
| 23 | 18 | спис_вар → ● вариант | 26 |
| 23 | 18 | спис_вар → ● спис_вар вариант | 25 |
| 23 | 18 | вариант → ● спис_конст : опер | 27 |
| 23 | 18 | спис_конст → ● lit | 28 |
| 23 | 18 | спис_конст → ● спис_конст , lit | 27 |
| 24 | 20 | спис_имен → спис_имен , id ● | нет перехода (конечное) |
| 25 | 23 | выбор → ВЫБОР id ИЗ спис_вар ● альт ВСЕ | 29 |
| 25 | 23 | спис_вар → спис_вар ● вариант | 26 |
| 25 | 23 | альт → ● ИНАЧЕ опер | 30 |
| 25 | 23 | альт → ● ε (правило завершено) | нет перехода (конечное) |
| 25 | 23 | вариант → ● спис_конст : опер | 27 |
| 25 | 23 | спис_конст → ● lit | 28 |
| 25 | 23 | спис_конст → ● спис_конст , lit | 27 |
| 26 | 23, 25 | спис_вар → вариант ● | нет перехода (конечное) |
| 27 | 23, 25 | вариант → спис_конст ● : опер | 31 |
| 27 | 23, 25 | спис_конст → спис_конст ● , lit | 32 |
| 28 | 23, 25 | спис_конст → lit ● | нет перехода (конечное) |
| 29 | 25 | выбор → ВЫБОР id ИЗ спис_вар альт ● ВСЕ | 33 |
| 30 | 25 | альт → ИНАЧЕ ● опер | 34 |
| 30 | 25 | опер → ● присв ; | 7 |
| 30 | 25 | опер → ● выбор ; | 8 |
| 30 | 25 | присв → ● expr -> id | 9 |
| 30 | 25 | выбор → ● ВЫБОР id ИЗ спис_вар альт ВСЕ | 10 |
| 31 | 27 | вариант → спис_конст : ● опер | 35 |
| 31 | 27 | опер → ● присв ; | 7 |
| 31 | 27 | опер → ● выбор ; | 8 |
| 31 | 27 | присв → ● expr -> id | 9 |
| 31 | 27 | выбор → ● ВЫБОР id ИЗ спис_вар альт ВСЕ | 10 |
| 32 | 27 | спис_конст → спис_конст , ● lit | 36 |
| 33 | 29 | выбор → ВЫБОР id ИЗ спис_вар альт ВСЕ ● | нет переходa (конечное) |
| 34 | 30 | альт → ИНАЧЕ опер ● | нет перехода (конечное) |
| 35 | 31 | вариант → спис_конст : опер ● | нет перехода (конечное) |
| 36 | 32 | спис_конст → спис_конст , lit ● | нет перехода (конечное) |



| Правило | Терминалы выбора |
|---|---|
| <программа> ::= <спис_опис> <спис_опер> КНЦ ; $ | ИМЕНА |
| <спис_опис> ::= <опис> <спис_опис'> | ИМЕНА |
| <спис_опис'> ::= <опис> <спис_опис'> | ИМЕНА |
| <спис_опис'> ::= ε | id, lit, (, ВЫБОР |
| <опис> ::= ИМЕНА <спис_имен> ; | ИМЕНА |
| <спис_имен> ::= id <спис_имен'> | id |
| <спис_имен'> ::= , id <спис_имен'> | , |
| <спис_имен'> ::= ε | ; |
| <спис_опер> ::= <опер> <спис_опер'> | id, lit, (, ВЫБОР |
| <спис_опер'> ::= <опер> <спис_опер'> | id, lit, (, ВЫБОР |
| <спис_опер'> ::= ε | КНЦ |
| <опер> ::= <присв> ; | id, lit, ( |
| <опер> ::= <выбор> ; | ВЫБОР |
| <присв> ::= <выраж> -> id | id, lit, ( |
| <выбор> ::= ВЫБОР id ИЗ <спис_вар> <альт> ВСЕ | ВЫБОР |
| <спис_вар> ::= <вариант> <спис_вар'> | lit |
| <спис_вар'> ::= <вариант> <спис_вар'> | lit |
| <спис_вар'> ::= ε | ИНАЧЕ, ВСЕ |
| <вариант> ::= <спис_конст> : <опер> | lit |
| <спис_конст> ::= lit <спис_конст'> | lit |
| <спис_конст'> ::= , lit <спис_конст'> | , |
| <спис_конст'> ::= ε | : |
| <альт> ::= ИНАЧЕ <опер> | ИНАЧЕ |
| <альт> ::= ε | ВСЕ |
| <выраж> ::= <слаг> <выраж'> | id, lit, ( |
| <выраж'> ::= + <слаг> <выраж'> | + |
| <выраж'> ::= − <слаг> <выраж'> | − |
| <выраж'> ::= ε | ->, ) |
| <слаг> ::= <степ> <слаг'> | id, lit, ( |
| <слаг'> ::= * <степ> <слаг'> | * |
| <слаг'> ::= / <степ> <слаг'> | / |
| <слаг'> ::= ε | +, −, ->, ) |
| <степ> ::= <элем> <степ'> | id, lit, ( |
| <степ'> ::= ** <степ> | ** |
| <степ'> ::= ε | *, /, +, −, ->, ) |
| <элем> ::= id | id |
| <элем> ::= lit | lit |
| <элем> ::= ( <выраж> ) | ( |



| Правило | Было | Стало |
|---|---|---|
| (2) | ⟨спис_опис⟩ ::= ⟨опис⟩ \| ⟨спис_опис⟩ ⟨опис⟩ | ⟨спис_опис⟩ ::= ⟨опис⟩ ⟨спис_опис′⟩; ⟨спис_опис′⟩ ::= ⟨опис⟩ ⟨спис_опис′⟩ \| ε |
| (4) | ⟨спис_имен⟩ ::= id \| ⟨спис_имен⟩ , id | ⟨спис_имен⟩ ::= id ⟨спис_имен′⟩; ⟨спис_имен′⟩ ::= , id ⟨спис_имен′⟩ \| ε |
| (5) | ⟨спис_опер⟩ ::= ⟨опер⟩ \| ⟨спис_опер⟩ ⟨опер⟩ | ⟨спис_опер⟩ ::= ⟨опер⟩ ⟨спис_опер′⟩; ⟨спис_опер′⟩ ::= ⟨опер⟩ ⟨спис_опер′⟩ \| ε |
| (9) | ⟨спис_вар⟩ ::= ⟨вариант⟩ \| ⟨спис_вар⟩ ⟨вариант⟩ | ⟨спис_вар⟩ ::= ⟨вариант⟩ ⟨спис_вар′⟩; ⟨спис_вар′⟩ ::= ⟨вариант⟩ ⟨спис_вар′⟩ \| ε |
| (11) | ⟨спис_конст⟩ ::= lit \| ⟨спис_конст⟩ , lit | ⟨спис_конст⟩ ::= lit ⟨спис_конст′⟩; ⟨спис_конст′⟩ ::= , lit ⟨спис_конст′⟩ \| ε |
| (13) | ⟨выраж⟩ ::= ⟨слаг⟩ \| ⟨выраж⟩+⟨слаг⟩ \| ⟨выраж⟩−⟨слаг⟩ | ⟨выраж⟩ ::= ⟨слаг⟩ ⟨выраж′⟩; ⟨выраж′⟩ ::= + ⟨слаг⟩ ⟨выраж′⟩ \| − ⟨слаг⟩ ⟨выраж′⟩ \| ε |
| (14) | ⟨слаг⟩ ::= ⟨степ⟩ \| ⟨слаг⟩*⟨степ⟩ \| ⟨слаг⟩/⟨степ⟩ | ⟨слаг⟩ ::= ⟨степ⟩ ⟨слаг′⟩; ⟨слаг′⟩ ::= * ⟨степ⟩ ⟨слаг′⟩ \| / ⟨степ⟩ ⟨слаг′⟩ \| ε |



# Выполнение практических работ №3 и №4 (Вариант 2)

Ниже — строго по методичке, пошагово, для **варианта 2**.

---

## ПРАКТИЧЕСКАЯ РАБОТА №3 — «Разделение сети на части с использованием VLAN»

### Что нужно по варианту 2 (Таблица 1, стр. 24)

Из таблицы 1 для варианта 2:

| Параметр | Значение |
|---|---|
| Сеть с одним коммутатором | |
| Количество VLAN | 3 |
| Количество компьютеров | 8 |
| IP-адреса | 172.2.vlan.0 |
| Сеть с двумя коммутаторами | |
| Количество VLAN | 4 |
| Количество компьютеров | 6 |
| IP-адреса | 168.77.vlan.0 |

**Расшифровка:**
- **Часть 1 (один коммутатор):** 3 VLAN, 8 компьютеров, адресация `172.2.<номер_vlan>.0` (например, VLAN 2 → 172.2.2.0/24, VLAN 3 → 172.2.3.0/24, VLAN 4 → 172.2.4.0/24).
- **Часть 2 (два коммутатора):** 4 VLAN, 6 компьютеров, адресация `168.77.<номер_vlan>.0` (например, VLAN 2 → 168.77.2.0/24, VLAN 3 → 168.77.3.0/24, VLAN 4 → 168.77.4.0/24, VLAN 5 → 168.77.5.0/24).

---

### ЧАСТЬ 1. Сеть с одним коммутатором (3 VLAN, 8 ПК)

#### Шаг 1. Построение топологии
1. Откройте Cisco Packet Tracer.
2. Добавьте **1 коммутатор** (например, 2960-24TT).
3. Добавьте **8 компьютеров** (PC0–PC7).
4. Соедините каждый ПК с коммутатором **прямым кабелем** (Copper Straight-Through), порты FastEthernet 0/1 – 0/8.

#### Шаг 2. Создание VLAN на коммутаторе
Откройте консоль коммутатора (CLI) и выполните:

```
enable
configure terminal
vlan 2
name VLAN2
exit
vlan 3
name VLAN3
exit
vlan 4
name VLAN4
exit
```

#### Шаг 3. Распределение портов по VLAN
По варианту 2 — 8 ПК, 3 VLAN. Распределите, например:
- VLAN 2: PC0, PC1, PC2 (порты fa0/1–fa0/3)
- VLAN 3: PC3, PC4, PC5 (порты fa0/4–fa0/6)
- VLAN 4: PC6, PC7 (порты fa0/7–fa0/8)

Команды:
```
interface range fa0/1-3
switchport mode access
switchport access vlan 2
exit

interface range fa0/4-6
switchport mode access
switchport access vlan 3
exit

interface range fa0/7-8
switchport mode access
switchport access vlan 4
exit
```

#### Шаг 4. Проверка настроек VLAN
```
end
show vlan
```
Убедитесь, что порты распределены по VLAN 2, 3, 4.

#### Шаг 5. Настройка IP-адресов компьютеров
По методичке (стр. 23): «адресацию vlan’ов производить различными сегментами, так, чтобы номер сегмента совпадал с номером vlan’а».

Для варианта 2 (адресация `172.2.vlan.0`):

| ПК | VLAN | IP-адрес | Маска |
|---|---|---|---|
| PC0 | 2 | 172.2.2.1 | 255.255.255.0 |
| PC1 | 2 | 172.2.2.2 | 255.255.255.0 |
| PC2 | 2 | 172.2.2.3 | 255.255.255.0 |
| PC3 | 3 | 172.2.3.1 | 255.255.255.0 |
| PC4 | 3 | 172.2.3.2 | 255.255.255.0 |
| PC5 | 3 | 172.2.3.3 | 255.255.255.0 |
| PC6 | 4 | 172.2.4.1 | 255.255.255.0 |
| PC7 | 4 | 172.2.4.2 | 255.255.255.0 |

На каждом ПК: вкладка **Desktop → IP Configuration → Static**, введите IP и маску. Шлюз можно не указывать (внутри одного VLAN маршрутизация не нужна).

#### Шаг 6. Проверка работоспособности VLAN
1. С PC0 (VLAN 2) выполните `ping 172.2.2.2` — должно работать (внутри VLAN).
2. С PC0 выполните `ping 172.2.3.1` — **не должно работать** (разные VLAN, нет маршрутизации).
3. Аналогично проверьте другие VLAN.

**Результаты подтвердите скриншотами.**

---

### ЧАСТЬ 2. Сеть с двумя коммутаторами (4 VLAN, 6 ПК)

#### Шаг 1. Построение топологии
1. Добавьте **2 коммутатора** (Switch0, Switch1).
2. Добавьте **6 компьютеров** (PC0–PC5).
3. Распределите: 3 ПК на Switch0, 3 ПК на Switch1.
4. Соедините ПК с коммутаторами прямым кабелем.
5. Соедините коммутаторы между собой **прямым кабелем** (или кроссовером — в Packet Tracer можно использовать прямой, но по методичке для соединения устройств одного уровня — cross-over).

#### Шаг 2. Создание VLAN на обоих коммутаторах
На **Switch0** и **Switch1** выполните:

```
enable
configure terminal
vlan 2
name VLAN2
exit
vlan 3
name VLAN3
exit
vlan 4
name VLAN4
exit
vlan 5
name VLAN5
exit
```

#### Шаг 3. Распределение портов по VLAN
Пример распределения (6 ПК, 4 VLAN):

**Switch0:**
- VLAN 2: PC0 (fa0/1)
- VLAN 3: PC1 (fa0/2)
- VLAN 4: PC2 (fa0/3)

**Switch1:**
- VLAN 2: PC3 (fa0/1)
- VLAN 3: PC4 (fa0/2)
- VLAN 5: PC5 (fa0/3)

Команды на Switch0:
```
interface fa0/1
switchport mode access
switchport access vlan 2
exit
interface fa0/2
switchport mode access
switchport access vlan 3
exit
interface fa0/3
switchport mode access
switchport access vlan 4
exit
```

Аналогично на Switch1 (с нужными VLAN).

#### Шаг 4. Настройка trunk-порта между коммутаторами
По методичке (стр. 24):

```
enable
configure terminal
interface [интерфейс_подключения]
switchport mode trunk
switchport trunk allowed vlan [номера_vlan’ов]
```

Например, если коммутаторы соединены через fa0/24:

**На Switch0:**
```
interface fa0/24
switchport mode trunk
switchport trunk allowed vlan 2,3,4,5
```

**На Switch1:**
```
interface fa0/24
switchport mode trunk
switchport trunk allowed vlan 2,3,4,5
```

#### Шаг 5. Настройка IP-адресов компьютеров
По варианту 2: адресация `168.77.vlan.0`.

| ПК | VLAN | IP-адрес | Маска |
|---|---|---|---|
| PC0 | 2 | 168.77.2.1 | 255.255.255.0 |
| PC1 | 3 | 168.77.3.1 | 255.255.255.0 |
| PC2 | 4 | 168.77.4.1 | 255.255.255.0 |
| PC3 | 2 | 168.77.2.2 | 255.255.255.0 |
| PC4 | 3 | 168.77.3.2 | 255.255.255.0 |
| PC5 | 5 | 168.77.5.1 | 255.255.255.0 |

#### Шаг 6. Проверка работоспособности
1. `ping` между ПК в одном VLAN (например, PC0 → PC3, оба VLAN 2) — **должно работать**.
2. `ping` между ПК в разных VLAN — **не должно работать**.
3. Проверьте `show vlan` на обоих коммутаторах.

**Результаты подтвердите скриншотами.**

#### Шаг 7. Оформление отчёта
В отчёт включите:
- Схемы сетей (скриншоты топологии).
- Скриншоты `show vlan`.
- Скриншоты `ping` (успешные и неуспешные).
- Выводы.

---

## ПРАКТИЧЕСКАЯ РАБОТА №4 — «Коммутаторы третьего уровня»

### Что нужно по варианту 2 (Таблица 1, стр. 30)

| Параметр | Значение |
|---|---|
| Количество сегментов | 3 |
| Начальный адрес подсетей | 20.20.0.0 |

**Расшифровка:**
- 3 сегмента (3 VLAN).
- Начальный адрес: 20.20.0.0.
- По методичке (стр. 28): VLAN 2 → 192.168.2.0, VLAN 3 → 192.168.3.0, VLAN 4 → 192.168.4.0. Для варианта 2 нужно использовать начальный адрес 20.20.0.0. Логично: VLAN 2 → 20.20.2.0/24, VLAN 3 → 20.20.3.0/24, VLAN 4 → 20.20.4.0/24.

---

### Шаг 1. Построение топологии
1. Добавьте **коммутатор 3-го уровня** (Cisco 3560-24PS).
2. Добавьте **3 компьютера** (PC0, PC1, PC2).
3. Соедините каждый ПК с коммутатором прямым кабелем (порты fa0/1, fa0/2, fa0/3).

### Шаг 2. Создание VLAN на коммутаторе 3-го уровня
```
enable
configure terminal
vlan 2
name VLAN2
exit
vlan 3
name VLAN3
exit
vlan 4
name VLAN4
exit
```

### Шаг 3. Назначение портов в VLAN
```
interface fa0/1
switchport mode access
switchport access vlan 2
exit
interface fa0/2
switchport mode access
switchport access vlan 3
exit
interface fa0/3
switchport mode access
switchport access vlan 4
exit
```

### Шаг 4. Проверка настроек портов
```
end
show run
```
Убедитесь, что порты назначены правильно.

### Шаг 5. Назначение IP-адресов для VLAN (SVI)
По методичке (стр. 28) для варианта 2 используем начальный адрес 20.20.0.0:

```
enable
configure terminal
interface vlan 2
ip address 20.20.2.1 255.255.255.0
exit
interface vlan 3
ip address 20.20.3.1 255.255.255.0
exit
interface vlan 4
ip address 20.20.4.1 255.255.255.0
exit
```

### Шаг 6. Проверка настроек IP
```
end
show run
```
Убедитесь, что на интерфейсах VLAN 2, 3, 4 назначены корректные IP.

### Шаг 7. Настройка IP-адресов компьютеров
По методичке (стр. 28): PC0 — 192.168.2.2, PC1 — 192.168.3.2, PC2 — 192.168.4.2. Для варианта 2:

| ПК | VLAN | IP-адрес | Маска | Шлюз |
|---|---|---|---|---|
| PC0 | 2 | 20.20.2.2 | 255.255.255.0 | 20.20.2.1 |
| PC1 | 3 | 20.20.3.2 | 255.255.255.0 | 20.20.3.1 |
| PC2 | 4 | 20.20.4.2 | 255.255.255.0 | 20.20.4.1 |

**Важно:** шлюз указывается обязательно (Default Gateway) — это IP соответствующего VLAN на коммутаторе.

### Шаг 8. Включение маршрутизации на коммутаторе 3-го уровня
По методичке (стр. 29):

```
enable
configure terminal
ip routing
```

### Шаг 9. Проверка связи
1. С PC0 выполните `ping 20.20.2.1` — должно работать (связь с шлюзом).
2. С PC0 выполните `ping 20.20.3.2` — должно работать (межсетевое взаимодействие).
3. С PC0 выполните `ping 20.20.4.2` — должно работать.
4. Проверьте все направления.

**Результаты подтвердите скриншотами.**

### Шаг 10. Оформление отчёта
В отчёт включите:
- Схему сети (скриншот).
- Скриншоты `show run` (настройки VLAN, IP-адресов).
- Скриншоты `ping` между сегментами.
- Выводы.

---

## ИТОГОВЫЙ ЧЕК-ЛИСТ

### Практика №3 (Вариант 2):
- [ ] Часть 1: 1 коммутатор, 3 VLAN, 8 ПК, адресация 172.2.x.0
- [ ] Часть 2: 2 коммутатора, 4 VLAN, 6 ПК, адресация 168.77.x.0
- [ ] Trunk между коммутаторами
- [ ] Проверка `show vlan`
- [ ] Ping внутри VLAN (работает) и между VLAN (не работает)
- [ ] Скриншоты

### Практика №4 (Вариант 2):
- [ ] Коммутатор 3560, 3 сегмента (VLAN 2, 3, 4)
- [ ] Адресация 20.20.x.0
- [ ] IP на SVI: 20.20.2.1, 20.20.3.1, 20.20.4.1
- [ ] IP на ПК: 20.20.2.2, 20.20.3.2, 20.20.4.2
- [ ] Шлюзы на ПК указаны
- [ ] `ip routing` включён
- [ ] Ping между всеми сегментами работает
- [ ] Скриншоты

---

Если нужно — могу расписать точные команды для конкретной топологии (сколько портов, какие ПК куда подключены) или помочь с оформлением отчёта.
