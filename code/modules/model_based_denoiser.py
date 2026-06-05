
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 


class ParameterModule(nn.Module):
    def __init__(self, initial_value, requires_grad=True):
        super(ParameterModule, self).__init__()
        self.param = nn.Parameter(torch.tensor(initial_value, dtype=torch.float), requires_grad=requires_grad)

    def forward(self):
        return self.param       

class DU_denoiser(nn.Module):   
    def __init__(self, DnCnn, blocks):
        super(DU_denoiser, self).__init__()
        self.Denoiser = DnCnn
        self.blocks = blocks
        self.channels = 64
        self.S = self.downsampling_operator(2)
        
        self.horizontal_gradient = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 25), padding=(0, 12), bias=False)
        self.horizontal_gradient_transposed = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 25), padding=(0, 12), bias=False)

        self.vertical_gradient = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(25, 1), padding=(12, 0), bias=False)
        self.vertical_gradient_transposed = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(25, 1), padding=(12, 0), bias=False)

        self.a1 = ParameterModule(0.09)
        self.a2 = ParameterModule(0.09)
        self.a3 = ParameterModule(0.09)
        self.a4 = ParameterModule(0.09)
        self.a5 = ParameterModule(0.09)


    def forward(self, data):
        device = data.device
        N = data.shape[3]
        self.S = self.S.to(device)
        S = self.S.unsqueeze(0).expand(data.shape[0], -1, -1)

        # X = F.interpolate(S@data, size=(self.channels, N), mode='bilinear', align_corners=True)
        X = copy.deepcopy(data)
        # X = self.Denoiser(data)
        
        for kk in range(self.blocks):
            
            grad_h = self.horizontal_gradient(X)
            grad_h_t = self.horizontal_gradient_transposed(grad_h)

            grad_v = self.vertical_gradient(X)
            grad_v_t = self.vertical_gradient_transposed(grad_v)

            X = self.a1.param*X + self.a2.param*data + self.a3.param* self.Denoiser(X,1) - self.a4.param*grad_v_t - self.a5.param*grad_h_t 
    
        return X


# class Super_Resolution(nn.Module):
#     def __init__(self, DnCnn, blocks):
#         super(Super_Resolution, self).__init__()
#         self.Denoiser = DnCnn
#         self.beta = ParameterModule(0.1)
#         self.alpha = ParameterModule(0.1)
#         self.iterations = blocks
#         self.channels = 64  # 64
#         self.S = self.downsampling_operator(4)
#         self.Thres = 4
#         self.upper_thres = 60  # Upper threshold

#         self.register_buffer = self.S
#         self.attention = LightweightMaskGatedAttention(1)
    
#     def forward(self, data, weighted_masked_labels):
#         device = data.device
#         batch_size = data.shape[0]
#         N = data.shape[3]
#         self.S = self.S.to(device)

#         self.DtD_base = torch.linalg.inv(
#                 self.beta.param*torch.eye(self.S.shape[1], device=device) + 
#                 (self.S.T @ self.S) + 1e-19 * torch.eye(self.S.shape[1]).to(device)
#             )
#         DtD = self.DtD_base.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

#         S = self.S.unsqueeze(0).expand(data.shape[0], -1, -1).to(device)
#         A = torch.matmul(S.transpose(1, 2).unsqueeze(dim=1), data) 

#         rec_range_image = F.interpolate(data, size=(self.channels, N), mode='bilinear', align_corners=True)
#         center_mask = (rec_range_image < self.Thres | (rec_range_image > self.upper_thres)).float()
#         rec_range_image = rec_range_image * (1 - center_mask)  
        
#         # Iterative refinement loop
#         for kk in range(self.iterations):
#             #Data concistency

#             x = torch.matmul(DtD, (A + self.beta.param * rec_range_image))   
#             center_mask_x = (x < self.Thres  | (x > self.upper_thres) | torch.isnan(x)).float()
#             x = x * (1 - center_mask_x)  

            
#             #Denoiser
#             rec_range_image = self.Denoiser(x, center_mask)
            
#             center_mask = (rec_range_image < self.Thres  | (rec_range_image > self.upper_thres) | torch.isnan(rec_range_image)).float()
#             rec_range_image = rec_range_image * (1 - center_mask)  

#             rec_range_image = self.attention(rec_range_image, weighted_masked_labels)

        
#         return  rec_range_image


    def downsampling_operator(self, s):
        
        S = np.zeros((int(self.channels/s), self.channels))
        vector = np.zeros((1, self.channels))
        vector[0, 0] = 1
        for i in range(int(self.channels/s)):
            S[i, :] = vector
            vector = np.roll(vector, s, axis=1)
        return torch.from_numpy(S).float()


























