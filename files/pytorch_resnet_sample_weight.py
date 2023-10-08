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

# print(len(train_loader))

# import models.resnet9 
from models.resnet import *
from models.resnet9new2 import resnet9jssc
from models.resnet_adc import resnet20_adc
# -------------------------------------------------------------------------------

print('Building model ...')
model = resnet20_adc()
print(model)

model.load_state_dict(torch.load('model_resnet_adc.pt'))

# print(model.conv1)
print(model.conv1.weight.max(), model.conv1.weight.min())
print(model.layer1[0].conv1.weight.max(), model.layer1[0].conv1.weight.min())
print(model.layer1[0].conv2.weight.max(), model.layer1[0].conv2.weight.min())
print(model.layer1[1].conv1.weight.max(), model.layer1[1].conv1.weight.min())
print(model.layer1[1].conv2.weight.max(), model.layer1[1].conv2.weight.min())
print(model.layer1[2].conv1.weight.max(), model.layer1[2].conv1.weight.min())
print(model.layer1[2].conv2.weight.max(), model.layer1[2].conv2.weight.min())
print(model.layer2[0].conv1.weight.max(), model.layer2[0].conv1.weight.min())
print(model.layer2[0].conv2.weight.max(), model.layer2[0].conv2.weight.min())
print(model.layer2[1].conv1.weight.max(), model.layer2[1].conv1.weight.min())
print(model.layer2[1].conv2.weight.max(), model.layer2[1].conv2.weight.min())
print(model.layer2[2].conv1.weight.max(), model.layer2[2].conv1.weight.min())
print(model.layer2[2].conv2.weight.max(), model.layer2[2].conv2.weight.min())
print(model.layer3[0].conv1.weight.max(), model.layer3[0].conv1.weight.min())
print(model.layer3[0].conv2.weight.max(), model.layer3[0].conv2.weight.min())
print(model.layer3[1].conv1.weight.max(), model.layer3[1].conv1.weight.min())
print(model.layer3[1].conv2.weight.max(), model.layer3[1].conv2.weight.min())
print(model.layer3[2].conv1.weight.max(), model.layer3[2].conv1.weight.min())
print(model.layer3[2].conv2.weight.max(), model.layer3[2].conv2.weight.min())
print(model.linear.weight.max(), model.linear.weight.min())
