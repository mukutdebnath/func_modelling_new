import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
import pdb
from quant_dorefa import *
__all__ = ['net']

class resnet(nn.Module):

    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):
        # print('Input:', torch.mean(x))

        x1 = self.fq0(x)

        # print('Conv1_In:', torch.mean(x1))

        out = self.conv1(x1)
        # print('Conv1: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn1(out)
        out = self.relu1(out)
        residual = out.clone() 

        # print('Conv1_out: ', torch.mean(out))
        # pdb.set_trace()

        out = self.fq1(out)
        # print('Conv2_In: ', torch.mean(out))
        # pdb.set_trace()

        out = self.conv2(out)
        # print('Conv2: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn2(out)
        out = self.relu2(out)

        out = self.fq2(out)
        out = self.conv3(out)
        # print('Conv3: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn3(out)
        out+=residual
        out = self.relu3(out)
        residual = out.clone() 
        
        ################################### 
        out = self.fq3(out)
        out = self.conv4(out)
        # print('Conv4: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn4(out)
        out = self.relu4(out)

        out = self.fq4(out)
        out = self.conv5(out)
        # print('Conv5: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn5(out)
        out+=residual
        out = self.relu5(out)
        residual = out.clone() 
        
        ################################### 
        out = self.fq5(out)
        out = self.conv6(out)
        # print('Conv6: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn6(out)
        out = self.relu6(out)

        out = self.fq6(out)
        out = self.conv7(out)
        # print('Conv7: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn7(out)
        out+=residual
        out = self.relu7(out)
        residual = out.clone() 
        
        ################################### 
        out = self.fq7(out)
        out = self.conv8(out)
        # print('Conv8: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn8(out)
        out = self.relu8(out)

        out = self.fq8(out)
        out = self.conv9(out)
        # print('Conv9: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn9(out)

        r1 = self.fqr1(residual)
        residual = self.resconv1(r1)

        out+=residual
        out = self.relu9(out)
        residual = out.clone() 
        
        ################################### 
        out = self.fq9(out)
        out = self.conv10(out)
        # print('Conv10: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn10(out)
        out = self.relu10(out)

        out = self.fq10(out)
        out = self.conv11(out)
        # print('Conv11: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn11(out)
        out+=residual
        out = self.relu11(out)
        residual = out.clone() 

        ################################### 
        out = self.fq11(out)
        out = self.conv12(out)
        # print('Conv12: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn12(out)
        out = self.relu12(out)

        out = self.fq12(out)
        out = self.conv13(out)
        # print('Conv13: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn13(out)
        out+=residual
        out = self.relu13(out)
        residual = out.clone() 
        
        ################################### 
        out = self.fq13(out)
        out = self.conv14(out)
        # print('Conv14: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn14(out)
        out = self.relu14(out)

        out = self.fq14(out)
        out = self.conv15(out)
        # print('Conv15: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn15(out)

        r2 = self.fqr2(residual)
        residual = self.resconv2(r2)

        out+=residual
        out = self.relu15(out)
        residual = out.clone() 

        ################################### 
        out = self.fq15(out)
        out = self.conv16(out)
        # print('Conv16: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn16(out)
        out = self.relu16(out)

        out = self.fq16(out)
        out = self.conv17(out)
        # print('Conv17: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn17(out)
        out+=residual
        out = self.relu17(out)
        residual = out.clone() 

        ################################### 
        out = self.fq17(out)
        out = self.conv18(out)
        # print('Conv18: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn18(out)
        out = self.relu18(out)

        out = self.fq18(out)
        out = self.conv19(out)
        # print('Conv19: ', torch.mean(out))
        # pdb.set_trace()

        out = self.bn19(out)
        out+=residual
        out = self.relu19(out)
        residual = out.clone() 
        
        ################################### 
        #########Layer################ 
        x=out
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn20(x)

        x = self.fq19(x)
        x = self.fc(x)
        
        # print('FC: ', torch.mean(out))
        # pdb.set_trace()

        x = self.bn21(x)
        x = self.logsoftmax(x)
        return x

class ResNet_cifar(resnet):

    def __init__(self, num_classes=100, a_bit=7, af_bit=4, w_bit=7, wf_bit=6):
        super(ResNet_cifar, self).__init__()

        
        self.abit = a_bit
        self.af_bit = af_bit
        self.wbit = w_bit
        self.wf_bit = wf_bit


        QConv2d = conv2d_Q_fn(w_bit=self.wbit, wf_bit=self.wf_bit)
        QLinear = linear_Q_fn(w_bit=self.wbit, wf_bit=self.wf_bit)
        

        self.inflate = 1
        self.fq0 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.conv1=QConv2d(3,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1= nn.BatchNorm2d(16*self.inflate)
        self.relu1=nn.ReLU(inplace=True)
        self.fq1 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv2=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2= nn.BatchNorm2d(16*self.inflate)
        self.relu2=nn.ReLU(inplace=True)
        self.fq2 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv3=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3= nn.BatchNorm2d(16*self.inflate)
        self.relu3=nn.ReLU(inplace=True)
        self.fq3 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        self.conv4=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4= nn.BatchNorm2d(16*self.inflate)
        self.relu4=nn.ReLU(inplace=True)
        self.fq4 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv5=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5= nn.BatchNorm2d(16*self.inflate)
        self.relu5=nn.ReLU(inplace=True)
        self.fq5 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        self.conv6=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6= nn.BatchNorm2d(16*self.inflate)
        self.relu6=nn.ReLU(inplace=True)
        self.fq6 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv7=QConv2d(16*self.inflate,16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7= nn.BatchNorm2d(16*self.inflate)
        self.relu7=nn.ReLU(inplace=True)
        self.fq7 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        #########Layer################ 
        self.conv8=QConv2d(16*self.inflate,32*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn8= nn.BatchNorm2d(32*self.inflate)
        self.relu8=nn.ReLU(inplace=True)
        self.fq8 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv9=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9= nn.BatchNorm2d(32*self.inflate)
        self.fqr1 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.resconv1=nn.Sequential(QConv2d(16*self.inflate,32*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
        nn.BatchNorm2d(32*self.inflate),)
        self.relu9=nn.ReLU(inplace=True)
        self.fq9 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        self.conv10=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10= nn.BatchNorm2d(32*self.inflate)
        self.relu10=nn.ReLU(inplace=True)
        self.fq10 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv11=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11= nn.BatchNorm2d(32*self.inflate)
        self.relu11=nn.ReLU(inplace=True)
        self.fq11 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        #######################################################

        self.conv12=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12= nn.BatchNorm2d(32*self.inflate)
        self.relu12=nn.ReLU(inplace=True)
        self.fq12 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv13=QConv2d(32*self.inflate,32*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13= nn.BatchNorm2d(32*self.inflate)
        self.relu13=nn.ReLU(inplace=True)
        self.fq13 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        #########Layer################ 
        self.conv14=QConv2d(32*self.inflate,64*self.inflate, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14= nn.BatchNorm2d(64*self.inflate)
        self.relu14=nn.ReLU(inplace=True)
        self.fq14 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv15=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15= nn.BatchNorm2d(64*self.inflate)
        self.fqr2 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        self.resconv2=nn.Sequential(QConv2d(32*self.inflate,64*self.inflate, kernel_size=1, stride=2, padding =0, bias=False),
        nn.BatchNorm2d(64*self.inflate),)
        self.relu15=nn.ReLU(inplace=True)
        self.fq15 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        self.conv16=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16= nn.BatchNorm2d(64*self.inflate)
        self.relu16=nn.ReLU(inplace=True)
        self.fq16 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv17=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17= nn.BatchNorm2d(64*self.inflate)
        self.relu17=nn.ReLU(inplace=True)
        self.fq17 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        self.conv18=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18= nn.BatchNorm2d(64*self.inflate)
        self.relu18=nn.ReLU(inplace=True)
        self.fq18 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.conv19=QConv2d(64*self.inflate,64*self.inflate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19= nn.BatchNorm2d(64*self.inflate)
        self.relu19=nn.ReLU(inplace=True)
        self.fq19 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)
        #######################################################

        #########Layer################ 
        self.avgpool=nn.AvgPool2d(8)
        self.bn20= nn.BatchNorm1d(64*self.inflate)
        self.fq20 = activation_quantize_fn(a_bit=self.abit, af_bit=self.af_bit)

        self.fc=QLinear(64*self.inflate,num_classes, bias=False)
        self.bn21= nn.BatchNorm1d(num_classes)
        self.logsoftmax=nn.LogSoftmax(dim=1)


def net(**kwargs):
    num_classes, depth, dataset, a_bit, af_bit,w_bit, wf_bit = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'a_bit', 'af_bit', 'w_bit', 'wf_bit'])
    return ResNet_cifar(num_classes=num_classes, a_bit=a_bit, af_bit=af_bit, w_bit=w_bit, wf_bit=wf_bit)