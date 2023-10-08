import random
import pandas
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from scipy import interpolate

from MyWorks.dataInterpols import InterpolFSS, InterpolFSSMC, tSS, tSSMC, adc_in_max_SS, adc_in_min_SS
from MyWorks.dataInterpols import InterpolFSSRamp, tSSRamp, adc_in_max_SSRamp, adc_in_min_SSRamp

# SS ADC ---------------------------------------------------------------------------------------------------------------

class SSADCFunction(Function):

    @staticmethod
    def forward(ctx, input, relu_range, adc_bits):
        result = input.clone()
        result = torch.div(result, relu_range)
        result = torch.mul(result, adc_in_max_SS)
        result = torch.clamp(result, min=adc_in_min_SS, max=adc_in_max_SS)
        result = result.cpu()
        result = InterpolFSSMC(result)
        result = input.new(result)
        result = result.cuda()
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.div(result, tSSMC)
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}'.format(input.max(), result.max()))
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class CCOADCFunction(Function):
    @staticmethod
    def forward(ctx, input, relu_range, adc_bits):
        data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/CCO_char_all_corners.csv')
        adc_in_min = data['Iin'].min()
        adc_in_max = data['Iin'].max()
        adc_out_max = data['Nom'].max()
        InterpolF = interpolate.interp1d(data['Iin'], data['Nom'])

        result = input.clone()
        result = torch.div(result, relu_range)
        result = torch.mul(result, adc_in_max)
        result = torch.clamp(result, min=adc_in_min, max=adc_in_max)
        result = result.cpu()
        result = InterpolF(result)
        result = input.new(result)
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.div(result, adc_out_max)
        result = torch.floor(result)
        # print(result.max())
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
        result = input.clone().detach()
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
    def __init__(self, relu_range, adc_bits):
        super(ADC, self).__init__()    
        # self.relu_range = relu_range  
        self.relu_range = relu_range
        self.adc_bits = adc_bits 
     
    def forward(self, x):
        return CCOADCFunction.apply(x, self.relu_range, self.adc_bits)
        # breakpoint()
        # if (self.adc == 'ss'):
        #     return SSADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), self.adc_bits)
        # elif (self.adc == 'cco'):
        #     return CCOADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), self.adc_bits)
        # else:
        #     return IdealADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), self.adc_bits)
    
# CCO ADC ---------------------------------------------------------------------------------------------------------------


    
class CCOADC(nn.Module):
    def __init__(self, adc_bits):
        super(CCOADC, self).__init__()
        # self.relu_range = relu_range
        self.adc_bits = adc_bits

    def forward(self, x):
        return CCOADCFunction.apply(x, x.reshape(x.shape[0],-1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3), self.adc_bits)


# CCO ADC MC randomized---------------------------------------------------------------------------------------------------------------

class CCOADCFunction_MC(Function):
    @staticmethod
    def forward(ctx, input, relu_range, adc_bits, pool_index):
        data = pandas.read_csv('/home/dynamo/a/debnathm/func_modelling/puma_functional_model/Simdata/Montecarlo_for_CCO_10u.csv')
        
        temp_data = data[data['mc_iteration']==pool_index+1]
        InterpolF = interpolate.interp1d(temp_data['i'], temp_data['FREQ'])
        
        adc_in_min = data['i'].min()
        adc_in_max = data['i'].max()
        adc_out_max = temp_data['FREQ'].max()
        # adc_out_max = data['FREQ'].max()

        result = input.clone().detach()
        result = torch.div(result, relu_range)
        result = torch.mul(result, adc_in_max)
        result = torch.where(result < adc_in_min, adc_in_min, result)
        result = result.cpu()
        result = InterpolF(result)
        result = input.new(result)
        result = torch.mul(result, pow(2, adc_bits))
        result = torch.div(result, adc_out_max)
        result = torch.floor(result)
        result = torch.mul(result, relu_range)
        result = torch.div(result, pow(2, adc_bits))
        # print('Input: {}, Output: {}'.format(input.max(), result.max()))
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
    
class CCOADC_MC(nn.Module):
    def __init__(self, output_size, relu_range, adc_bits):
        super(CCOADC_MC, self).__init__()
        self.relu_range = relu_range
        self.adc_bits = adc_bits
        self.output_size = output_size

    def forward(self, x):
        output = torch.empty_like(x)
        for i in range(self.output_size):
            pool_index = random.randint(0,199)
            output[:,i] = CCOADCFunction_MC.apply(x[:,i], self.relu_range, self.adc_bits, pool_index)            

        return output



# Ideal ADC ---------------------------------------------------------------------------------------------------------------
    


class IdealADC(nn.Module):    
    def __init__(self, relu_range, adc_bits):
        super(IdealADC, self).__init__()    
        self.relu_range = relu_range  
        self.adc_bits = adc_bits
     
    def forward(self, x):
        return IdealADCFunction.apply(x, self.relu_range, self.adc_bits)