import torch

# #range
# max = 8
# min = 4
# #create tensor with random values in range (min, max)
# rand_tensor = (max-min)*torch.rand((2, 5)) + min
# #print tensor
# print(rand_tensor)

# reference CCO_adc_10u_all_corners

scaling_factor_min = 0.85
scaling_factor_max = 1.15

tensor1 = (scaling_factor_max-scaling_factor_min) * torch.rand((1,16,32,32)) + scaling_factor_min
tensor2 = (scaling_factor_max-scaling_factor_min) * torch.rand((1,32,16,16)) + scaling_factor_min
tensor3 = (scaling_factor_max-scaling_factor_min) * torch.rand((1,64,8,8)) + scaling_factor_min

tensor_list = [tensor1, tensor2, tensor3]

torch.save(tensor_list, 'Variation_matrix_factor_CCO1.pt')