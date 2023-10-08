import pandas
import torch
import torch.nn as nn
from scipy import interpolate
from MyWorks.dataInterpols import InterpolF
  
class SS_adc(nn.Module):    
    def __init__(self):
        super(SS_adc, self).__init__()  
        self.InterpolF = InterpolF           
     
    def forward(self, x):
        return self.InterpolF(x)