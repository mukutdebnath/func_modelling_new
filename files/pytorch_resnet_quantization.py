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
best_accuracy = 0

# import models.resnet9 
from models.resnet import *
from models.resnet9new2 import resnet9jssc
from models.resnet_adc import resnet20_adc
# -------------------------------------------------------------------------------

print('Building model ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)

model = resnet20().to(device)
print(model)
print('Hyperparameters: Epochs: {}, Batch Size: {}, Learning Rate: {}'.format(num_epochs, batch_size, learning_rate))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 1e-4, momentum = 0.9) 
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=0 - 1) 

# Train the model
total_step = len(train_loader)

# weight quantization
def weight_quantization(weights):
    weight_bits = 5
    weight_range = 10    # weight range = -10 to +10
    new_weights = torch.add(weights, weight_range)
    new_weights = torch.div(new_weights, 2*weight_range)
    new_weights = torch.mul(new_weights, pow(2, weight_bits))
    new_weights = torch.floor(new_weights)
    new_weights = torch.mul(new_weights, 2*weight_range)
    new_weights = torch.div(new_weights, pow(2, weight_bits))
    new_weights = torch.sub(new_weights, weight_range)
    return new_weights

import gc
total_step = len(train_loader)

for epoch in range(num_epochs):

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

        conv1weight = weight_quantization(model.conv1.weight.clone())
        model.conv1.weight = nn.Parameter(weight_quantization(model.conv1.weight.clone()))

        model.layer1[0].conv1.weight = nn.Parameter(weight_quantization(model.layer1[0].conv1.weight.clone()))
        model.layer1[0].conv2.weight = nn.Parameter(weight_quantization(model.layer1[0].conv2.weight.clone()))
        model.layer1[1].conv1.weight = nn.Parameter(weight_quantization(model.layer1[1].conv1.weight.clone()))
        model.layer1[1].conv2.weight = nn.Parameter(weight_quantization(model.layer1[1].conv2.weight.clone()))
        model.layer1[2].conv1.weight = nn.Parameter(weight_quantization(model.layer1[2].conv1.weight.clone()))
        model.layer1[2].conv2.weight = nn.Parameter(weight_quantization(model.layer1[2].conv2.weight.clone()))

        model.layer2[0].conv1.weight = nn.Parameter(weight_quantization(model.layer2[0].conv1.weight.clone()))
        model.layer2[0].conv2.weight = nn.Parameter(weight_quantization(model.layer2[0].conv2.weight.clone()))
        model.layer2[1].conv1.weight = nn.Parameter(weight_quantization(model.layer2[1].conv1.weight.clone()))
        model.layer2[1].conv2.weight = nn.Parameter(weight_quantization(model.layer2[1].conv2.weight.clone()))
        model.layer2[2].conv1.weight = nn.Parameter(weight_quantization(model.layer2[2].conv1.weight.clone()))
        model.layer2[2].conv2.weight = nn.Parameter(weight_quantization(model.layer2[2].conv2.weight.clone()))

        model.layer3[0].conv1.weight = nn.Parameter(weight_quantization(model.layer3[0].conv1.weight.clone()))
        model.layer3[0].conv2.weight = nn.Parameter(weight_quantization(model.layer3[0].conv2.weight.clone()))
        model.layer3[1].conv1.weight = nn.Parameter(weight_quantization(model.layer3[1].conv1.weight.clone()))
        model.layer3[1].conv2.weight = nn.Parameter(weight_quantization(model.layer3[1].conv2.weight.clone()))
        model.layer3[2].conv1.weight = nn.Parameter(weight_quantization(model.layer3[2].conv1.weight.clone()))
        model.layer3[2].conv2.weight = nn.Parameter(weight_quantization(model.layer3[2].conv2.weight.clone()))

        # model.linear.weight = nn.parameter(weight_quantization(model.linear.weight.clone()))


        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    t1 = time.time()
    print ('Epoch [{}/{}], Current lr: {}, Loss: {:.4f}, Epoch time: {}.' 
                   .format(epoch+1, num_epochs, optimizer.param_groups[0]['lr'], loss.item(), t1- t0))

    lr_scheduler.step()

    # Validation
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
            torch.save(model.state_dict(), 'model_resnet_weightQ.pt')

        print('Accuracy of the network on the {} test images: {} %; Best Accuracy: {}'.format(10000, curr_accuracy, best_accuracy)) 

 
