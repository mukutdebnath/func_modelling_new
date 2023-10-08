'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from quant_dorefa import *
from MyWorks.classes import *

import src.config as cfg

if cfg.if_bit_slicing and not cfg.dataset:
    from src.pytorch_mvm_class_v3 import *
elif cfg.dataset:
    from geneix.pytorch_mvm_class_dataset import *   # import mvm class from geneix folder
else:
    from src.pytorch_mvm_class_no_bitslice import *

__all__ = ['ResNet', 'resnet20', 'resnet20_mvm', 'resnet20_adc', 'resnet20_adc_vars', 'resnet20_q', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

conv1relu=[]
layerconv1relu=[]
layerconv2relu=[]

# base fp model------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
# adc model fp -----------------------------------------

class BasicBlockADC(nn.Module):
    expansion = 1
    def __init__(self, adc, adc_bits, adc_func, adc_params, in_planes, planes, stride=1, option='A'):
        super(BasicBlockADC, self).__init__()
        self.adc = adc
        self.adc_bits = adc_bits
        self.adc_func = adc_func
        self.adc_params = adc_params
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.adcl1 = ADC(adc=self.adc, adc_bits=self.adc_bits, adc_func=self.adc_func, adc_params=self.adc_params)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.adcl2 = ADC(adc=self.adc, adc_bits=self.adc_bits, adc_func=self.adc_func, adc_params=self.adc_params)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.adcl1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.adcl2(out)
        return out


class ResNetADC(nn.Module):
    def __init__(self, adc, adc_bits, adc_func, adc_params, block, num_blocks, num_classes=10):
        super(ResNetADC, self).__init__()
        self.in_planes = 16
        self.adc = adc
        self.adc_bits = adc_bits
        self.adc_func = adc_func
        self.adc_params = adc_params
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.adc1 = ADC(adc=self.adc, adc_bits=self.adc_bits, adc_func=self.adc_func, adc_params=self.adc_params)
        self.layer1 = self._make_layer(block, adc=self.adc, adc_bits=self.adc_bits, 
                                       adc_func=self.adc_func, adc_params=self.adc_params, 
                                       planes=16, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, adc=self.adc, adc_bits=self.adc_bits, 
                                       adc_func=self.adc_func, adc_params=self.adc_params, 
                                       planes=32, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, adc=self.adc, adc_bits=self.adc_bits, 
                                       adc_func=self.adc_func, adc_params=self.adc_params, 
                                       planes=64, num_blocks=num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, adc, adc_bits, adc_func, adc_params, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(adc, adc_bits, adc_func, adc_params, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('--------------------------------')
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.adc1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print('--------------------------------')
        return out
    
# #   ----------------------------------------------------------------------------------------------------------------
    
class BasicBlockADCVar(nn.Module):
    expansion = 1

    def __init__(self, adc, adc_bits, adc_func, adc_params, sf_range, in_planes, planes, stride=1, option='A'):
        super(BasicBlockADCVar, self).__init__()
        self.adc = adc
        self.adc_bits = adc_bits
        self.adc_func = adc_func
        self.adc_params = adc_params
        self.sf_range = sf_range
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.adcl1 = ADC_VAR(adc=self.adc, adc_bits=self.adc_bits, adc_func=self.adc_func, adc_params=self.adc_params, sf_range=self.sf_range)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.adcl2 = ADC_VAR(adc=self.adc, adc_bits=self.adc_bits, adc_func=self.adc_func, adc_params=self.adc_params, sf_range=self.sf_range)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.adcl1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.adcl2(out)
        return out

class ResNetADCVar(nn.Module):
    def __init__(self, adc, adc_bits, adc_func, adc_params, sf_range, block, num_blocks, num_classes=10):
        super(ResNetADCVar, self).__init__()
        self.in_planes = 16
        self.adc = adc
        self.adc_bits = adc_bits
        self.adc_func = adc_func
        self.adc_params = adc_params
        self.sf_range = sf_range
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.adc1 = ADC_VAR(adc=self.adc, adc_bits=self.adc_bits, adc_func=self.adc_func, adc_params=self.adc_params, sf_range=self.sf_range)
        self.layer1 = self._make_layer(block, adc=self.adc, adc_bits=self.adc_bits, 
                                       adc_func=self.adc_func, adc_params=self.adc_params, sf_range=self.sf_range, 
                                       planes=16, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, adc=self.adc, adc_bits=self.adc_bits, 
                                       adc_func=self.adc_func, adc_params=self.adc_params, sf_range=self.sf_range, 
                                       planes=32, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, adc=self.adc, adc_bits=self.adc_bits, 
                                       adc_func=self.adc_func, adc_params=self.adc_params, sf_range=self.sf_range, 
                                       planes=64, num_blocks=num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, adc, adc_bits, adc_func, adc_params, sf_range, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(adc, adc_bits, adc_func, adc_params, sf_range, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.adc1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#---------------------------------------------------------------------------------------------------------------------------

class BasicBlockQ(nn.Module):
    expansion = 1

    def __init__(self, wbits, abits, in_planes, planes, stride=1, option='A'):
        super(BasicBlockQ, self).__init__()
        self.wbits = wbits
        self.abits = abits

        self.fq = activation_quantize_fn(a_bit=self.abits[0], af_bit=self.abits[1])
        QConv2d = conv2d_Q_fn(w_bit=self.wbits[0], wf_bit=self.wbits[1])

        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     QConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.fq(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.fq(out)
        return out


class ResNetQ(nn.Module):
    def __init__(self, wbits, abits, block, num_blocks, num_classes=10):
        super(ResNetQ, self).__init__()
        self.in_planes = 16
        self.wbits = wbits
        self.abits = abits

        self.fq = activation_quantize_fn(a_bit=self.abits[0], af_bit=self.abits[1])
        QConv2d = conv2d_Q_fn(w_bit=self.wbits[0], wf_bit=self.wbits[1])
        QLinear = linear_Q_fn(w_bit=self.wbits[0], wf_bit=self.wbits[1])

        self.conv1 = QConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, wbits=self.wbits, abits=self.abits,
                                       planes=16, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, wbits=self.wbits, abits=self.abits,
                                       planes=32, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, wbits=self.wbits,  abits=self.abits,
                                       planes=64, num_blocks=num_blocks[2], stride=2)
        self.linear = QLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, wbits, abits, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(wbits, abits, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('--------------------------------')
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.fq(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print('--------------------------------')
        return out
    
# #----------------------------------------------------------------------------------------------------------------

class BasicBlockMVM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockMVM, self).__init__()
        self.conv1 = Conv2d_mvm(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_mvm(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d_mvm(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetMVM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetMVM, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2d_mvm(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = Linear_mvm(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20():
    "Baseline floating point ResNet20"
    return ResNet(BasicBlock, [3, 3, 3])

def resnet20_mvm():
    "MVM functional ResNet20: parameters form src.config"
    return ResNetMVM(BasicBlockMVM, [3, 3, 3])

def resnet20_adc(adc, adc_bits, adc_func, adc_params):
    "Floating point ResNet20 with ADC tf after ReLUs"
    return ResNetADC(adc, adc_bits, adc_func, adc_params, BasicBlockADC, [3, 3, 3])

def resnet20_adc_vars(adc, adc_bits, adc_func, adc_params, sf_range):        # var model includes random variation across corners or MC simulations
    "Floating point ResNet20 with ADCs including Gaussian Noise to the ADC outputs"
    return ResNetADCVar(adc, adc_bits, adc_func, adc_params, sf_range, BasicBlockADCVar, [3, 3, 3])

def resnet20_q(wbits, abits):
    "Weight & Activation quantized ResNet20 without ADCs"
    return ResNetQ(wbits, abits, BasicBlockQ, [3, 3, 3])

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
