import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
import copy


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)

        attn_0 = self.conv1_1(attn)
        attn_0 = self.conv1_2(attn_0)

        attn_1 = self.conv2_1(attn)
        attn_1 = self.conv2_2(attn_1)

        attn_2 = self.conv3_1(attn)
        attn_2 = self.conv3_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv4(attn)

        return attn * x


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act =nn.LeakyReLU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.downsample = downsample

        self.conv = BasicConv2d(planes, planes, 3, padding=1)

        self.attn = SpatialAttention(planes)
        # self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        out = self.conv(x)
        out = out + (self.attn(out))

        out += x

        return out



class ResNet_denoiser(nn.Module):
    def __init__(self, nclasses, params=5, block=BasicBlock, layers=[2, 2, 2, 2], if_BN=False, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64, channels = [16, 32, 64]):
        super(ResNet_denoiser, self).__init__()
        self.nclasses = nclasses


        self.input_size = nclasses 
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.channels1 = channels[0]
        self.channels2 = channels[1]
        self.channels3 = channels[2]

        self.conv1 = BasicConv2d(self.input_size,  self.channels1, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(self.channels1, self.channels2, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(self.channels2, self.channels2, kernel_size=3, padding=1)

        self.inplanes = self.channels2

        self.layer1 = self._make_layer(block, self.channels2, layers[0])
        self.layer2 = self._make_layer(block, self.channels2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels2, layers[3], stride=2)

        self.decoder1 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)
        self.decoder2 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)
        self.decoder3 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)
        self.decoder4 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)

        self.fusion_conv = BasicConv2d(self.channels2 * 3, self.channels2, kernel_size=1)
        self.semantic_output = nn.Conv2d(self.channels2, nclasses, 1)

        # Center mask attention
        self.center_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    # nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    # SoftPool2d(kernel_size=(2, 2), stride=(2, 2)),
                    # norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                    # SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
                    # nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x_input, center_mask=1):
        # center_attention = self.center_attention(center_mask)

        x = self.conv1(x_input)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)    # 1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_1 = self.decoder1(torch.cat((x, x_1), dim=1))


        res_2 = F.interpolate(
            x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_2 = self.decoder2(torch.cat((res_1, res_2), dim=1))


        res_3 = F.interpolate(
            x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = self.decoder3(torch.cat((res_2, res_3), dim=1))


        res_4 = F.interpolate(
            x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = self.decoder4(torch.cat((res_3, res_4), dim=1))

      
        res = [res_2, res_3, res_4]
        
 

        out = torch.cat(res, dim=1)


        out = self.fusion_conv(out)

        out = (self.semantic_output(out)
               )

        return out
    

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation), delta=1)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False), delta=1)


# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
#         super(BasicConv2d, self).__init__()
#         self.relu = relu
#         self.conv = spectral_norm(nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False), delta=1)
#         self.bn = nn.BatchNorm2d(out_planes)
#         if self.relu:
#             self.relu = nn.LeakyReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.relu:
#             x = self.relu(x)
#         return x


# class AttentionModule(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = spectral_norm(nn.Conv2d(dim, dim, 5, padding=2, groups=dim), delta=1)

#         self.conv1_1 = spectral_norm(nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim), delta=1)
#         self.conv1_2 = spectral_norm(nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim), delta=1)

#         self.conv2_1 = spectral_norm(nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim), delta=1)
#         self.conv2_2 = spectral_norm(nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim), delta=1)

#         self.conv3_1 = spectral_norm(nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim), delta=1)
#         self.conv3_2 = spectral_norm(nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim), delta=1)

#         self.conv4 = spectral_norm(nn.Conv2d(dim, dim, 1), delta=1)

#     def forward(self, x):
#         attn = self.conv0(x)

#         attn_0 = self.conv1_1(attn)
#         attn_0 = self.conv1_2(attn_0)

#         attn_1 = self.conv2_1(attn)
#         attn_1 = self.conv2_2(attn_1)

#         attn_2 = self.conv3_1(attn)
#         attn_2 = self.conv3_2(attn_2)

#         attn = attn + attn_0 + attn_1 + attn_2

#         attn = self.conv4(attn)

#         return attn * x


# class SpatialAttention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.proj_1 = nn.Conv2d(dim, dim, 1)
#         self.act =nn.LeakyReLU()
#         self.spatial_gating_unit = AttentionModule(dim)
#         self.proj_2 = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         shortcut = x.clone()
#         x = self.proj_1(x)
#         x = self.act(x)
#         x = self.spatial_gating_unit(x)
#         x = self.proj_2(x)
#         x = x + shortcut
#         return x


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, if_BN=None):
#         super(BasicBlock, self).__init__()
#         if groups != 1 or base_width != 64:
#             raise ValueError(
#                 'BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError(
#                 "Dilation > 1 not supported in BasicBlock")

#         self.downsample = downsample

#         self.conv = BasicConv2d(planes, planes, 3, padding=1)

#         self.attn = SpatialAttention(planes)
#         # self.dropout = nn.Dropout2d(p=0.2)

#     def forward(self, x):
#         if self.downsample is not None:
#             x = self.downsample(x)

#         out = self.conv(x)
#         out = out + (self.attn(out))

#         out += x

#         return out



# class ResNet_denoiser(nn.Module):
#     def __init__(self, nclasses, params=5, block=BasicBlock, layers=[2, 2, 2, 2], if_BN=False, zero_init_residual=False,
#                  norm_layer=None, groups=1, width_per_group=64, channels = [16, 32, 64]):
#         super(ResNet_denoiser, self).__init__()
#         self.nclasses = nclasses


#         self.input_size = nclasses 
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.if_BN = if_BN
#         self.dilation = 1

#         self.groups = groups
#         self.base_width = width_per_group
#         self.channels1 = channels[0]
#         self.channels2 = channels[1]
#         self.channels3 = channels[2]

#         self.conv1 = BasicConv2d(self.input_size,  self.channels1, kernel_size=3, padding=1)
#         self.conv2 = BasicConv2d(self.channels1, self.channels2, kernel_size=3, padding=1)
#         self.conv3 = BasicConv2d(self.channels2, self.channels2, kernel_size=3, padding=1)

#         self.inplanes = self.channels2

#         self.layer1 = self._make_layer(block, self.channels2, layers[0])
#         self.layer2 = self._make_layer(block, self.channels2, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, self.channels2, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, self.channels2, layers[3], stride=2)

#         self.decoder1 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)
#         self.decoder2 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)
#         self.decoder3 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)
#         self.decoder4 = BasicConv2d(self.channels3, self.channels2, 3, padding=1)

#         self.fusion_conv = BasicConv2d(self.channels2 * 3, self.channels2, kernel_size=1)
#         self.semantic_output = spectral_norm(nn.Conv2d(self.channels2, nclasses, 1), delta=1)

#         # Center mask attention
#         self.center_attention = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if self.if_BN:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     # nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
#                     # SoftPool2d(kernel_size=(2, 2), stride=(2, 2)),
#                     # norm_layer(planes * block.expansion),
#                 )
#             else:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride)
#                     # SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
#                     # nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
#                 )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, if_BN=self.if_BN))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(planes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 if_BN=self.if_BN))

#         return nn.Sequential(*layers)

#     def forward(self, x_input, center_mask):
#         # center_attention = self.center_attention(center_mask)

#         x = self.conv1(x_input)
#         x = self.conv2(x)
#         x = self.conv3(x)

#         x_1 = self.layer1(x)    # 1
#         x_2 = self.layer2(x_1)  # 1/2
#         x_3 = self.layer3(x_2)  # 1/4
#         x_4 = self.layer4(x_3)  # 1/8

#         res_1 = self.decoder1(torch.cat((x, x_1), dim=1))


#         res_2 = F.interpolate(
#             x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
#         res_2 = self.decoder2(torch.cat((res_1, res_2), dim=1))


#         res_3 = F.interpolate(
#             x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
#         res_3 = self.decoder3(torch.cat((res_2, res_3), dim=1))


#         res_4 = F.interpolate(
#             x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
#         res_4 = self.decoder4(torch.cat((res_3, res_4), dim=1))

      
#         res = [res_2, res_3, res_4]
        
 

#         out = torch.cat(res, dim=1)


#         out = self.fusion_conv(out)

#         out = (self.semantic_output(out)
#                )

#         return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = (nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1))
        self.conv2 = (nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        self.conv3 = (nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        # Save the feature maps for skip connections
        x1 = F.relu(self.conv1(x))  
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        return x3, x2, x1

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convtrans1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x3, x2, x1):
        x = F.relu(self.convtrans1(x3))
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = F.relu(self.convtrans2(x))
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.convtrans3(x)
        return x

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x3, x2, x1 = self.encoder(x)
        out = self.decoder(x3, x2, x1)
        return out

    












from enum import Enum, auto

import torch
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
from torch.nn import functional as F

from typing import Optional

__all__ = ['orthogonal', 'spectral_norm']


def _is_orthogonal(Q, eps=None):
    n, k = Q.size(-2), Q.size(-1)
    Id = torch.eye(k, dtype=Q.dtype, device=Q.device)
    # A reasonable eps, but not too large
    eps = 10. * n * torch.finfo(Q.dtype).eps
    return torch.allclose(Q.mH @ Q, Id, atol=eps)


def _make_orthogonal(A):
    """ Assume that A is a tall matrix.
    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative
    """
    X, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)
    # The diagonal of X is the diagonal of R (which is always real) so we normalise by its signs
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q


class _OrthMaps(Enum):
    matrix_exp = auto()
    cayley = auto()
    householder = auto()


class _Orthogonal(Module):
    base: Tensor

    def __init__(self,
                 weight,
                 orthogonal_map: _OrthMaps,
                 *,
                 use_trivialization=True) -> None:
        super().__init__()

        if weight.is_complex() and orthogonal_map == _OrthMaps.householder:
            raise ValueError("The householder parametrization does not support complex tensors.")

        self.shape = weight.shape
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer("base", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n, k = X.size(-2), X.size(-1)
        transposed = n < k
        if transposed:
            X = X.mT
            n, k = k, n
        # Here n > k and X is a tall matrix
        if self.orthogonal_map == _OrthMaps.matrix_exp or self.orthogonal_map == _OrthMaps.cayley:
            # We just need n x k - k(k-1)/2 parameters
            X = X.tril()
            if n != k:
                # Embed into a square matrix
                X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
            A = X - X.mH
            # A is skew-symmetric (or skew-hermitian)
            if self.orthogonal_map == _OrthMaps.matrix_exp:
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == _OrthMaps.cayley:
                # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(n, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
            # Q is now orthogonal (or unitary) of size (..., n, n)
            if n != k:
                Q = Q[..., :k]
            # Q is now the size of the X (albeit perhaps transposed)
        else:
            # X is real here, as we do not support householder with complex numbers
            A = X.tril(diagonal=-1)
            tau = 2. / (1. + (A * A).sum(dim=-2))
            Q = torch.linalg.householder_product(A, tau)
            # The diagonal of X is 1's and -1's
            # We do not want to differentiate through this or update the diagonal of X hence the casting
            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)

        if hasattr(self, "base"):
            Q = self.base @ Q
        if transposed:
            Q = Q.mT
        return Q

    @torch.autograd.no_grad()
    def right_inverse(self, Q: torch.Tensor) -> torch.Tensor:
        if Q.shape != self.shape:
            raise ValueError(f"Expected a matrix or batch of matrices of shape {self.shape}. "
                             f"Got a tensor of shape {Q.shape}.")

        Q_init = Q
        n, k = Q.size(-2), Q.size(-1)
        transpose = n < k
        if transpose:
            Q = Q.mT
            n, k = k, n

        # We always make sure to always copy Q in every path
        if not hasattr(self, "base"):

            if self.orthogonal_map == _OrthMaps.cayley or self.orthogonal_map == _OrthMaps.matrix_exp:
                raise NotImplementedError("It is not possible to assign to the matrix exponential "
                                          "or the Cayley parametrizations when use_trivialization=False.")

            A, tau = torch.geqrf(Q)
            A.diagonal(dim1=-2, dim2=-1).sign_()
            # Equality with zero is ok because LAPACK returns exactly zero when it does not want
            # to use a particular reflection
            A.diagonal(dim1=-2, dim2=-1)[tau == 0.] *= -1
            return A.mT if transpose else A
        else:
            if n == k:
                # We check whether Q is orthogonal
                if not _is_orthogonal(Q):
                    Q = _make_orthogonal(Q)
                else:  # Is orthogonal
                    Q = Q.clone()
            else:
                # Complete Q into a full n x n orthogonal matrix
                N = torch.randn(*(Q.size()[:-2] + (n, n - k)), dtype=Q.dtype, device=Q.device)
                Q = torch.cat([Q, N], dim=-1)
                Q = _make_orthogonal(Q)
            self.base = Q

            neg_Id = torch.zeros_like(Q_init)
            neg_Id.diagonal(dim1=-2, dim2=-1).fill_(-1.)
            return neg_Id

def orthogonal(module: Module,
               name: str = 'weight',
               orthogonal_map: Optional[str] = None,
               *,
               use_trivialization: bool = True) -> Module:

    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    # We could implement this for 1-dim tensors as the maps on the sphere
    # but I believe it'd bite more people than it'd help
    if weight.ndim < 2:
        raise ValueError("Expected a matrix or batch of matrices. "
                         f"Got a tensor of {weight.ndim} dimensions.")

    if orthogonal_map is None:
        orthogonal_map = "matrix_exp" if weight.size(-2) == weight.size(-1) or weight.is_complex() else "householder"

    orth_enum = getattr(_OrthMaps, orthogonal_map, None)
    if orth_enum is None:
        raise ValueError('orthogonal_map has to be one of "matrix_exp", "cayley", "householder". '
                         f'Got: {orthogonal_map}')
    orth = _Orthogonal(weight,
                       orth_enum,
                       use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, name, orth, unsafe=True)
    return module



class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        delta: float = 0.5
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        self.delta = delta
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer('_u', F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer('_v', F.normalize(v, dim=0, eps=self.eps))

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:

        assert weight_mat.ndim > 1

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = F.normalize(torch.mv(weight_mat, self._v),      # type: ignore[has-type]
                                  dim=0, eps=self.eps, out=self._u)   # type: ignore[has-type]
            self._v = F.normalize(torch.mv(weight_mat.t(), self._u),
                                  dim=0, eps=self.eps, out=self._v)   # type: ignore[has-type]

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            return (weight / sigma)*self.delta


    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


def spectral_norm(module: Module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None, delta = 0.001) -> Module:

    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(module, name, _SpectralNorm(weight, n_power_iterations, dim, eps, delta))
    return module