import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
  """This functions returns a class of uniform quantization

  Args: 
    k : bit precision of quantization 

  Returns:
    A class with uniform quantization with backward and forward functions.
  """
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      # elif k == 1:
      #   out = torch.sign(input)
      else:
        n = float(2**k)
        out = torch.floor(input*n)/n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply

class weight_noise_fn(nn.Module):
  def __init__(self, gamma, w_bits, wf_bits):
    super(weight_noise_fn, self).__init__()
    self.gamma = gamma
    self.w_bits = w_bits
    self.wf_bits = wf_bits

  def forward(self, x):  # x are quantized weights, (7,7) say
    weight = x
    weight_int = torch.mul(weight, (2**self.wf_bits)).to(torch.int64)
    
    binary_comps_sq = torch.zeros_like(weight_int, dtype=torch.int64)
    
    i = 1
    while weight_int.max() > 0:
      if i > 2 ** self.w_bits:
        raise Exception('Weight magnitude exceeds maximum allowed.')
      else:
        binary_rep = i * (weight_int % 2)
        binary_comps_sq += binary_rep ** 2
        weight_int //= 2
        i *= 2

    binary_comps_sq_sqrt = torch.sqrt(binary_comps_sq)
    sigma = self.gamma * binary_comps_sq_sqrt / (2 ** self.wf_bits)

    # normal noise
    noise_n = torch.randn_like(x)
    # 0, sigma noise
    noise = noise_n * sigma    
    return noise
  

class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit, wf_bit):
    """This class quantization weights
    
    Args:
      a_bit(int)  : bit precision for weight quantization

    Returns:
      Class object for weight quantization 
    """
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 16 or w_bit == 32
    self.w_bit = w_bit
    self.wf_bit = wf_bit
    self.wi_bit = w_bit - wf_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    """This is a function that quantizes the weights 

        Args:
            x(tensor)    : weights data to be quantized

        Returns:
            Returns quantized tensor
        """   
    if self.w_bit == 32:
      weight_q = x
    # elif self.w_bit == 1:
    #   E = torch.mean(torch.abs(x)).detach()
    #   weight_q = self.uniform_q(x / E) * E
    else:
      

      ## weight = torch.tanh(x)

      # max_w = torch.max(torch.abs(weight)).detach()
      # weight = weight / 2 / max_w + 0.5
      # weight_q = max_w * (2 * self.uniform_q(weight) - 1)

      # weight = weight/2 + 0.5
      ## weight = torch.clamp(x, -1, 1-1/n)

      weight = torch.clamp(x, -2**self.wi_bit, 2**self.wi_bit-1/2**self.wf_bit)

      weight = weight/2**self.wi_bit

      # weight_q = (2 * self.uniform_q(weight) - 1)
      weight_q = self.uniform_q(weight)

      weight_q = weight_q*2**self.wi_bit

    return weight_q


class activation_quantize_fn(nn.Module):
  
  def __init__(self, a_bit, af_bit):
    """This class quantization activations
    
    Args:
      a_bit(int)  : bit precision for activation quantization

    Returns:
      Class object for activation quantization 
    """
  
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 16 or a_bit == 32
    self.a_bit = a_bit
    self.af_bit = af_bit
    self.ai_bit = a_bit - af_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    """This is a function that quantizes the activation 

        Args:
            x(tensor)    : activation data to be quantized

        Returns:
            Returns quantized tensor
        """   
        
    if self.a_bit == 32:
      activation_q = x
    else:
      # n = float(2**self.a_bit)

      activation = torch.clamp(x, -2**self.ai_bit, 2**self.ai_bit-1/2**self.af_bit)

      activation = activation/2**self.ai_bit

      # weight_q = (2 * self.uniform_q(weight) - 1)
      activation_q = self.uniform_q(activation)

      activation_q = activation_q*2**self.ai_bit

      # print(np.unique(activation_q.detach().numpy()))

    return activation_q


def conv2d_Q_fn(w_bit, wf_bit, gamma):
  """This is a function returns a conv2d Layer class which has quantized weights

    Args:
      w_bits (int)  : bit precision for weight quantization

    Returns:
      A class for conv2d with weight quantization. Parameters of the class same as torch.nn.Conv2d
        """    
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.wf_bit= wf_bit
      self.gamma = gamma
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit, wf_bit=wf_bit)
      self.noise_fn = weight_noise_fn(gamma, w_bit, wf_bit)


    def forward(self, input, order=None):  
      self.weight_q = self.quantize_fn(self.weight) 
      self.weight_q += self.noise_fn(self.weight_q)
      
      # self.weight.data = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, self.weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit, wf_bit, gamma):
  """This is a function returns a Linear Layer class which has quantized weights

    Args:
      w_bits (int)  : bit precision for weight quantization

    Returns:
      A class for fully connected layer with weight quantization.Parameters of the class same as torch.nn.Linear
        """   
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.wf_bit = wf_bit
      self.gamma = gamma
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit, wf_bit=wf_bit)
      self.noise_fn = weight_noise_fn(gamma, w_bit, wf_bit)

    def forward(self, input):
      self.weight_q = self.quantize_fn(self.weight) 
      self.weight_q += self.noise_fn(self.weight_q)
      # self.weight.data = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, self.weight_q, self.bias)

  return Linear_Q


if __name__ == '__main__':
  # import numpy as np
  # import matplotlib.pyplot as plt

  # a = torch.rand(1, 3, 32, 32)

  # Conv2d = conv2d_Q_fn(w_bit=1)
  # conv = Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

  # img = torch.randn(1, 256, 56, 56)
  # print(img.max().item(), img.min().item())
  # out = conv(img)
  # print(out.max().item(), out.min().item())

  a = torch.tensor(((0.5,0.25),(0.75, 1.0)))
  wbits=4
  wfbits=2

  wnf=weight_noise_fn(0.1, wbits, wfbits)

  print(a)
  print(wnf(a))