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

num_classes = 10
num_epochs = 300
batch_size = 128
learning_rate = 0.1
valid_size = 0.1

# dataset loading ----------------------------------------------------------

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

transform = transforms.Compose([
    # transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    Cutout(n_holes = 1, length = 16)
    ])

train_dataset = datasets.CIFAR10(root=datasets_dir, train=True,
                                transform=transform, target_transform=None, download=True)

test_dataset = datasets.CIFAR10(root=datasets_dir, train=False,
          download=True, transform=transform)

# breakpoint()

num_train = len(train_dataset)
train_idx = list(range(num_train))

np.random.seed(42)
np.random.shuffle(train_idx)
train_sampler = SubsetRandomSampler(train_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

print(len(train_loader))
print(len(test_loader))

# import models.resnet9 
from models.resnet import *
from models.resnet9new2 import resnet9jssc
from models.resnet9new2_adc import resnet9jssc_adc
from models.resnet_adc import resnet20_adc
# -------------------------------------------------------------------------------

print('Building model ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)

model = resnet9jssc_adc().to(device)
print(model)
print('Hyperparameters: Epochs: {}, Batch Size: {}, Learning Rate: {}'.format(num_epochs, batch_size, learning_rate))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 1e-4, momentum = 0.9) 
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=0 - 1) 

# Train the model
total_step = len(train_loader)
best_accuracy = 0

import gc
total_step = len(train_loader)

for epoch in range(num_epochs):
    # if epoch%60 == 0 and epoch != 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr']*0.1
    # print('Current lr:')
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'])

    t0 = time.time()
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # print(images.size())
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    t1 = time.time()
    print ('Epoch [{}/{}], Current lr: {}, Loss: {:.4f}, Epoch time: {}.' 
                   .format(epoch+1, num_epochs, optimizer.param_groups[0]['lr'], loss.item(), t1- t0))

    lr_scheduler.step()

    # Validation
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

        curr_accuracy = 100 * correct / total
        if (best_accuracy < curr_accuracy):
            best_accuracy = curr_accuracy
            torch.save(model.state_dict(), 'model_resnet9_adc_baseline.pt')

        print('Accuracy of the network on the {} test images: {} %; Best Accuracy: {}'.format(10000, curr_accuracy, best_accuracy)) 


