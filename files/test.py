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

import argparse
import torch
import torch.nn as nn
import pandas
import numpy as np
from scipy import interpolate

from MyWorks.adccharacteristics import get_adc

parser = argparse.ArgumentParser()

parser.add_argument('--a', help='model architecture', default='resnet20_adc')
parser.add_argument('--adc', help='adc type: cco, ss, incktcco', default='ideal')
parser.add_argument('--adcbits', help='number of adc bits', default=7, type=int)
parser.add_argument('--adcindex', help='corner or mc index, 0 for nominal in all cases', default=0, type=int)
parser.add_argument('--inp', default = 5e-6, type=float)
parser.add_argument('--wbits', default=[4,3], type=int)

args = parser.parse_args()

# print(get_adc(args.adc))
# adc_func, adc_min, adc_max, T =  get_adc(args.adc)
# print(adc_func(args.inp))
# print(adc_func(args.inp+1e-8))
# print(adc_func(args.inp))
# # print(adc_func(args.inp+1e-8))

# print(args.wbits[0], args.wbits[1])
# from models.resnet import resnet20

# model = resnet20()
# loaded_model=torch.load('resnet20_baseline.pt')
# model_dict = model.state_dict()

# breakpoint()

# for name, param in loaded_model.items():
#     print(name)

# for name, param in model_dict.items():
#     print(name)

# for name, param in loaded_model.items():
#     name_sliced = name[7:]
#     if name_sliced in model_dict:
#         model_dict[name_sliced].copy_(param)

# model.load_state_dict(model_dict)

# breakpoint()



# class Net(nn.Module):
#     def __init__(self, in_channels):
#         super(Net,self).__init__()
#         self.in_channels=in_channels
#         self.conv1=nn.Conv2d(in_channels=self.in_channels, out_channels=3, kernel_size=3,stride=1,padding=1)
        
#     def forward(self,x):
#         out = self.conv1(x)
#         return out
    
# model1=Net(1)
# model2=Net(2)

# breakpoint()

import torch
import torch.nn as nn

class ADCActivation(nn.Module):
    def __init__(self, adc_characteristics):
        super(ADCActivation, self).__init__()
        self.adc_characteristics = torch.tensor(list(adc_characteristics.values())).unsqueeze(0)

    def forward(self, x):
        thresholds = self.adc_characteristics.to(x.device)                               # threshold has shape [1, #adc levels]
        quantized_output = torch.sum(x.unsqueeze(-1) >= thresholds, dim=-1).float()      # x.unsqueeze(-1) has shape [shape, 1]
                                                    # the comparison gives output of shape  [shape, #adc_levles]
                                                    # sum is taken of last dimension i.e., -1 which gives the number of times the 
                                                        # comparison output is True, which gives the countin integer,
                                                        # convert it to float.
        return quantized_output

adc_characteristics = {
    'threshold0': 0.1,
    'threshold1': 0.3,
    'threshold2': 0.5,
    'threshold3': 0.7,
    'threshold4': 0.9,
    'threshold5': 1.1,
    'threshold6': 1.3,
    'threshold7': 1.5
}

adcnet = ADCActivation(adc_characteristics=adc_characteristics)
a=torch.tensor(((0.2,0.3),(0.8,0.5),(0.6,0.5)))

breakpoint()