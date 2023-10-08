import random
import pandas
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from scipy import interpolate
from quant_dorefa import activation_quantize_fn

# from MyWorks.dataInterpols import InterpolFSS, InterpolFSSMC, tSS, tSSMC, adc_in_max_SS, adc_in_min_SS
# from MyWorks.dataInterpols import InterpolFSSRamp, tSSRamp, adc_in_max_SSRamp, adc_in_min_SSRamp

# SS ADC ---------------------------------------------------------------------------------------------------------------

class SSADCFunction(Function):

    @staticmethod
    def forward(ctx, input, relu_range, adc_bits, adc_func, adc_params):
        result = input.clone()
        relu_range = torch.where(relu_range==0.0, 1.0, relu_range)
        result = torch.div(result, relu_range)
        result = torch.mul(result, adc_params[1])
        result = torch.clamp(result, min=adc_params[0], max=adc_params[1])
        result = result.cpu()
        # breakpoint()
        result = adc_func(result)
        result = input.new(result)
        result = result.cuda()
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.div(result, adc_params[2])
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}, Relu Range: {}'.format(input.min(), result.min(), relu_range.min()))
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class CCOADCFunction(Function):
    @staticmethod
    def forward(ctx, input, relu_range, adc_bits, adc_func, adc_params):
        result = input.clone()
        relu_range = torch.where(relu_range==0.0, 1.0, relu_range)
        result = torch.div(result, relu_range)
        result = torch.mul(result, adc_params[1])
        result = torch.clamp(result, min=adc_params[0], max=adc_params[1])
        # print(result.shape)
        result = result.cpu()
        result = adc_func(result)
        result = input.new(result)
        result = result.cuda()
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.div(result, adc_params[2])
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}'.format(input.max(), result.max()))
        return input.new(result)        

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class IdealADCFunction(Function):

    @staticmethod
    def forward(ctx, input, relu_range, adc_bits):
        result = input.clone()
        result = torch.div(result, relu_range)
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}'.format(input.max(), result.max()))
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
    
# ----------------------------------------------------------------------------------------------------------

class ADC(nn.Module):    
    def __init__(self, adc, adc_bits, adc_func, adc_params):
        super(ADC, self).__init__()    
        # self.relu_range = relu_range  
        self.adc = adc
        self.adc_bits = adc_bits 
        self.adc_func = adc_func
        self.adc_params = adc_params
     
    def forward(self, x):
        # breakpoint()
        if (self.adc == 'ss'):
            return SSADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), 
                                       self.adc_bits, self.adc_func, self.adc_params)
        elif (self.adc == 'cco'):
            return CCOADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), 
                                        self.adc_bits, self.adc_func, self.adc_params)
        elif (self.adc == 'ideal'):
            return IdealADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), 
                                          self.adc_bits)
        else: return x
    
# Ending of normal ADC---------------------------------------------------------------------------------------------------------------

class SSADCVARFunction(Function):

    @staticmethod
    def forward(ctx, input, relu_range, adc_bits, adc_func, adc_params, sf_range):
        result = input.clone()
        relu_range = torch.where(relu_range==0.0, 1.0, relu_range)
        result = torch.div(result, relu_range)
        result = torch.mul(result, adc_params[1])
        result = torch.clamp(result, min=adc_params[0], max=adc_params[1])
        result = result.cpu()
        # breakpoint()
        result = adc_func(result)
        result = input.new(result)
        result = result.cuda()
        # var_matrix = (sf_range[0]-sf_range[1]) * torch.rand(result.shape) + sf_range[1]
        var_matrix = torch.normal(mean=sf_range[0], std=sf_range[1], size=result.shape)
        var_matrix = var_matrix.cuda()
        result = torch.mul(result, var_matrix)
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.div(result, adc_params[2])
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}, Relu Range: {}'.format(input.min(), result.min(), relu_range.min()))
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None

class CCOADCVARFunction(Function):
    @staticmethod
    def forward(ctx, input, relu_range, adc_bits, adc_func, adc_params, sf_range):
        result = input.clone()
        relu_range = torch.where(relu_range==0.0, 1.0, relu_range)
        result = torch.div(result, relu_range)
        result = torch.mul(result, adc_params[1])
        result = torch.clamp(result, min=adc_params[0], max=adc_params[1])
        result = result.cpu()
        result = adc_func(result)
        result = input.new(result)
        result = result.cuda()
        # var_matrix = (sf_range[0]-sf_range[1]) * torch.rand(result.shape) + sf_range[1]
        var_matrix = torch.normal(mean=sf_range[0], std=sf_range[1], size=result.shape)
        var_matrix = var_matrix.cuda()
        result = torch.mul(result, var_matrix)
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.div(result, adc_params[2])
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}, VarMatrixMax: {}'.format(input.max(), result.max(), torch.mean(var_matrix)))
        return input.new(result)        

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None

class IdealADCVARFunction(Function):

    @staticmethod
    def forward(ctx, input, relu_range, adc_bits):
        result = input.clone()
        result = torch.div(result, relu_range)
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}'.format(input.max(), result.max()))
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
    
# ----------------------------------------------------------------------------------------------------------

class ADC_VAR(nn.Module):    
    def __init__(self, adc, adc_bits, adc_func, adc_params, sf_range):
        super(ADC_VAR, self).__init__()    
        # self.relu_range = relu_range  
        self.adc = adc
        self.adc_bits = adc_bits 
        self.adc_func = adc_func
        self.adc_params = adc_params
        self.sf_range = sf_range
     
    def forward(self, x):
        # breakpoint()
        if (self.adc == 'ss'):
            return SSADCVARFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), 
                                       self.adc_bits, self.adc_func, self.adc_params, self.sf_range)
        elif (self.adc == 'cco'):
            return CCOADCVARFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), 
                                        self.adc_bits, self.adc_func, self.adc_params, self.sf_range)
        elif (self.adc == 'ideal'):
            return IdealADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), 
                                          self.adc_bits)
        else: return x