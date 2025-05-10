#=======================================
# Ejemplo de red neuronal convolucional
#=======================================
# Traducido de pytorch tutorial 2023
#=======================================
# Chan Campos Ashanty Iyari
# Fundamentos de IA
# ESFM IPN Abril 2025
#=======================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from A14_TorchRedPrealimentada import train_dataset, test_loader

#========================
# Configuración del GPU
#========================
device == torch.device('cuda' if torch.cuda.is_aviable() else 'cpu')

#==================
# Hiper-parámetros
#==================
num_epochs = 10          # Iteraciiones sobre los datos de entrenamiento
batch_size = 4           # Subconjuntos de datos
learning_rate = 0.01     # Tasa se aprendizaje

#============================================================
# Definir pre-procesamiento de datos (transformación)
#============================================================
# La base de datos tiene imágenes PILImage en el rango [0,1].
# Las transformamos a Tensores de rango normalizado [-1,1]
#============================================================
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#=================================================================================
# CIFAR10: 60000 32x32 imágenes a color en 10 clases, con 6000 imágenes por clase
#=================================================================================
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=False)

#=======================
# Objetos a clasificar
#=======================
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#==========================
# Graficar con matplotlib
#==========================
def imsgow(img):
    img = img =2 + 0.5   #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#========================================
# Obtener algunas imágenes para entrenar
#========================================
dataiter = iter(train_loader)
images, labels = next(dataiter)

#===============================
# Mostrar contenido de imágenes
#===============================
imshow(torchvision.utils.make_grid(images))

#===========================
# Red neuronal convolucinal
#===========================
class ConvNet(nn.Module):
    def __init__(self:
        super(ConvNet, self).__init__()
                 # 3 entradas (a color), 6 salidas (filtros), 5x5 entradas en el Kernel de convolución
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Máximo de una ventana de 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # 6 entradas, 16 salidas (filtros), 5x5 entradas en el Kernel de convolución
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Redes lineales (entradas, salidas)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc1 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
    #maxpool( relu (convolución ) )
    x = self.pool(F.relu(self.conv1(x)))    # -> n, 6, 14, 14
    x = self.pool(F.relu(self.conv2(x)))    # -> n, 16, 5, 6
    # Reorganizar al tensor x
    x = x.view(-1, 16 * 5 * 5)              # -> n, 400
    # Redes lineales + relu
    x = F.relu(self.fc1(x))                  # -> n, 120
    x = F.relu(self.fc2(x))                  # -> n, 84
    # Red lineal
    x = self.fc3(x)
    return x

#=============================
# Correr el modelo en el GPU
#=============================
model = ConvNet().to(device)
#========================================================================
# Usar cross-entropy como costo y gradiente estocástico como optimizador
#========================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#=============================
# Iteraciones (entrenamiento)
#=============================
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Formato original: [4, 3, 32, 32] = 4, 3, 1024
        # Capa de entrada: 3 canales de entrada, 6 canales de salida, 5 tamaño del kernel

    # imágenes
    images = images.to(device)

    # etiquetas
    labels = labels.to(device)

    # Evaluación (forward pass)
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Gradiente y optimización (backward)
    #  inicializar gradiente a cero + calcularlo + aplicarlo
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 2000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
print('Entrenamiento completo')

#============================================
# Guardar resultado del modelo (parámetros)
#============================================
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

#==================
# Probar el modelo
#==================
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        otputs = model(images)
        # max regresa (valor, índice)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Precisión del modelo: {acc} %')

    for i in range(10):
        acc = 100.0 * n_correct[i] / n_class_samples[i]
        print(f'Precisión de {clases[i]}: {acc} %')

