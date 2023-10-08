import os
import sys
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(root_dir, "test")
src_dir = os.path.join(root_dir, "src")
models_dir = os.path.join(root_dir, "models")
datasets_dir = os.path.join(root_dir, "Datasets")

sys.path.insert(0, root_dir) # 1 adds path to end of PYTHONPATH
sys.path.insert(0, models_dir)
sys.path.insert(0, test_dir) 
sys.path.insert(0, src_dir)
sys.path.insert(0, datasets_dir)

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
num_epochs = 300
batch_size = 128
learning_rate = 0.1
valid_size = 0.1

# dataset loading ----------------------------------------------------------

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

train_dataset = datasets.CIFAR10(root=datasets_dir, train=True,
                                transform=transform, target_transform=None, download=True)

test_dataset = datasets.CIFAR10(root=datasets_dir, train=False,
          download=True, transform=transform)

# breakpoint()

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.seed(42)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler)
 
valid_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

# print(len(train_loader))

# import models.resnet9 
from models.resnet import *
from models.resnet9new2 import resnet9jssc
from models.resnet_adc import resnet20_adc
# -------------------------------------------------------------------------------

print('Building model ...')
model = resnet20_adc().to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 1e-4, momentum = 0.9)  

# Train the model
total_step = len(train_loader)

import gc
total_step = len(train_loader)

model.load_state_dict(torch.load('model_resnet_adc.pt'))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))  
