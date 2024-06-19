"""
Binary Convolution modules
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class binary_activation(nn.Module):
    def init(self):
        super(binary_activation, self).init()
    def forward(self, input_tensor):
        base_output = torch.sign(input_tensor)
        below_neg_one = input_tensor < -1
        below_zero = input_tensor < 0
        below_one = input_tensor < 1
        output_tag1 = (-1) * below_neg_one.float() + ((input_tensor ** 2) + 2 * input_tensor) * (1 - below_neg_one.float())
        output_tag2 = output_tag1 * below_zero.float() + ((-input_tensor ** 2) + 2 * input_tensor) * (1 - below_zero.float())
        output_tag3 = output_tag2 * below_one.float() + 1 * (1 - below_one.float())
        final_output = base_output.detach() - output_tag3.detach() + output_tag3
        return final_output

class learnable_bias(nn.Module):
    def init(self, out_channels):
        super(learnable_bias, self).init()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)
    def forward(self, input_tensor):
        output_tensor = input_tensor + self.bias.expand_as(input_tensor)
        return output_tensor


class hardbinary_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=True, bias=True, gamma=1.0, eps=1e-5):
        super(hardbinary_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(torch.mul(torch.rand((self.shape)), 0.001), requires_grad=True)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2

    def forward(self, x):
        float_weights = self.weight
        std, mean = torch.std_mean(float_weights, dim=[1, 2, 3], keepdim=True, unbiased=False)
        float_weights = self.scale * (float_weights - mean) / (std + self.eps)

        scaling_factor = torch.mean(torch.mean(torch.mean(abs(float_weights), dim=3, keepdim=True), dim=2, keepdim=True),dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()

        binary_weights_without_grad = scaling_factor * torch.sign(float_weights)
        cliped_binary_weights = torch.clamp(float_weights, -1.0, 1.0)
        binary_weights = binary_weights_without_grad.detach() - cliped_binary_weights.detach() + cliped_binary_weights 

        x = x.type_as(binary_weights)
        output_tensor = nn.functional.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        return output_tensor

class binary_conv(nn.Module):
    default_act = binary_activation()
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True):

        super().__init__()

        # self.add0 = (s == 1 and in_channels == out_channels)
        # self.add1 = (s == 2 and in_channels == out_channels)        
        # self.add2 = (s == 1 and in_channels != out_channels)    
        # self.add3 = (s == 2 and in_channels != out_channels and 2*in_channels != out_channels)
        # self.add4 = (s == 2 and 2*in_channels == out_channels )

        self.bias1 = learnable_bias(in_channels)
        self.conv0 = hardbinary_conv(in_channels, out_channels, k, s, autopad(k, p, d), groups=g, bias=False)
        self.bias2 = learnable_bias(out_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

        self.c = min(in_channels, out_channels)//16

        self.pool = nn.AvgPool2d(2, 2)
        
        # if self.add2 or self.add3:
        #   self.conv1 = nn.Sequential(nn.Conv2d(in_channels, self.c, 1, 1, 0, groups=g, dilation=d, bias=False), nn.BatchNorm2d(self.c))
        #   self.conv2 = nn.Sequential(nn.Conv2d(self.c, out_channels, 1, 1, 0, groups=g, dilation=d, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):

        out = self.bias1(x)
        out = self.act(out)
        out = self.conv0(out)
        # out = self.bn0(out)
        # if self.add0:
        #     out += x 
        #     out = self.bn0(out)  
        # if self.add1:
        #     out += self.pool(x)
        #     out = self.bn0(out)  
        # if self.add2:
        #     out = self.bn0(out)
        #     out += self.conv2(self.conv1(x))
        # if self.add3:
        #     out = self.bn0(out)
        #     out += self.conv2(self.conv1(self.pool(x)))
        # if self.add4:
        #     out += torch.cat([self.pool(x),self.pool(x)],dim=1)
        #     out = self.bn0(out)
        out = self.bias2(out)
        out = self.prelu(out)
        out = self.bias2(out)

        return out

    def forward_fuse(self, x):

        out = self.bias1(x)
        out = self.act(out)
        out = self.conv0(out) 
        # if self.add0:
        #     out += x                      
        # if self.add1:
        #     out += self.pool(x)                        
        # if self.add2:
        #     out += self.conv2(self.conv1(x))
        # if self.add3:
        #     out += self.conv2(self.conv1(self.pool(x)))
        # if self.add4:
        #     out += torch.cat([self.pool(x),self.pool(x)],dim=1)
        out = self.bias2(out)
        out = self.prelu(out)
        out = self.bias2(out)

        return out
'''
class anti_sign(nn.Module):
    def __init__(self):
        super(anti_sign, self).__init__()

    def forward(self, x, b):
        result = x.clone()
        lower_bound = torch.min(x) * 0.5
        upper_bound = torch.max(x) * 0.5
        mask_1 = x < lower_bound
        mask_2 = (x >= lower_bound) & (x < 0)
        mask_3 = (x >= 0) & (x < upper_bound)
        mask_4 = x >= upper_bound
        result[mask_1] = 0
        result[mask_2] = 1
        result[mask_3] = 2
        result[mask_4] = 3
        adjusted_result = result - b
        adjusted_result[adjusted_result == 1] = -1
        adjusted_result[adjusted_result == 2] = 1

        negative_mask = x < -1
        subzero_mask = x < 0
        positive_mask = x < 1
        tmp1 = (-1) * negative_mask.float() + (x * x + 2 * x) * (~negative_mask).float()
        tmp2 = tmp1 * subzero_mask.float() + (-x * x + 2 * x) * (~subzero_mask).float()
        final_result = tmp2 * positive_mask.float() + 1 * (~positive_mask).float()
        adjusted_result = adjusted_result.detach() - final_result.detach() + final_result
        
        return adjusted_result

class dupbinary_conv(nn.Module):
    default_act = binary_activation()
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True):
        super().__init__()
        self.bias1 = learnable_bias(in_channels)
        self.conv0 = hardbinary_conv(in_channels, out_channels, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv1 = hardbinary_conv(in_channels, out_channels, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bias11 = learnable_bias(out_channels)
        self.bias12 = learnable_bias(out_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.add0 = (s == 1 and in_channels == out_channels)
        # self.add1 = (s == 2 and in_channels == out_channels)
        # self.add2 = (s == 2 and 2 * in_channels == out_channels)
        # self.add3 = (s == 1 and in_channels != out_channels)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.anti_sign = anti_sign()

        self.prelu1 = nn.PReLU(out_channels)
        self.prelu2 = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.bias1(x)
        out1 = self.act(out)
        out2 = self.anti_sign(out, out1)

        out1 = self.conv0(out1)
        out1 = self.bn0(out1)

        # if self.add0:
        #     out1 += x
        # if self.add1:
        #     out1 = out1 + self.pool(x)
        # if self.add2:
        #     out1 = out1 + torch.cat([self.pool(x), self.pool(x)], 1)
        # if self.add3:
        #     sl = list(x.chunk(math.ceil(self.in_channels / self.out_channels), dim=1))
        #     for i in range(len(sl)):
        #         if out1.shape[1] != sl[i].shape[1]:
        #             sl[i] = F.pad(sl[i], (0, 0, 0, 0, 0, out1.shape[1]-sl[i].shape[1]))
        #         out1 = out1 + (1 / math.ceil(self.in_channels / self.out_channels)) * sl[i]

        out1 = self.bias11(out1)
        out1 = self.prelu1(out1)
        out1 = self.bias11(out1)

        out2 = self.conv1(out2)
        out2 = self.bn1(out2)

        # if self.add0:
        #     out2 += x
        # if self.add1:
        #     out2 = out2 + self.pool(x)
        # if self.add2:
        #     out2 = out2 + torch.cat([self.pool(x), self.pool(x)],1)
        # if self.add3:
        #     sl = list(x.chunk(math.ceil(self.in_channels / self.out_channels), dim=1))
        #     for i in range(len(sl)):
        #         if out2.shape[1] != sl[i].shape[1]:
        #             sl[i] = F.pad(sl[i], (0, 0, 0, 0, 0, out2.shape[1]-sl[i].shape[1]))
        #         out2 = out2 + (1 / math.ceil(self.in_channels / self.out_channels)) * sl[i]

        out2 = self.bias12(out2)
        out2 = self.prelu2(out2)
        out2 = self.bias12(out2)

        out3 = out1 + out2

        return out3

    def forward_fuse(self, x):
        out = self.bias1(x)
        out1 = self.act(out)
        out2 = self.anti_sign(out, out1)

        out1 = self.conv0(out1)

        # if self.add0:
        #     out1 += x
        # if self.add1:
        #     out1 = out1 + self.pool(x)
        # if self.add2:
        #     out1 = out1 + torch.cat([self.pool(x), self.pool(x)],1)
        # if self.add3:
        #     sl = list(x.chunk(math.ceil(self.in_channels / self.out_channels), dim=1))
        #     for i in range(len(sl)):
        #         if out1.shape[1] != sl[i].shape[1]:
        #             sl[i] = F.pad(sl[i], (0, 0, 0, 0, 0, out1.shape[1]-sl[i].shape[1]))
        #         out1 = out1 + (1 / math.ceil(self.in_channels / self.out_channels)) * sl[i]

        out1 = self.bias11(out1)
        out1 = self.prelu1(out1)
        out1 = self.bias11(out1)

        out2 = self.conv1(out2)

        # if self.add0:
        #     out2 += x
        # if self.add1:
        #     out2 = out2 + self.pool(x)
        # if self.add2:
        #     out2 = out2 + torch.cat([self.pool(x), self.pool(x)],1)
        # if self.add3:
        #     sl = list(x.chunk(math.ceil(self.in_channels / self.out_channels), dim=1))
        #     for i in range(len(sl)):
        #         if out2.shape[1] != sl[i].shape[1]:
        #             sl[i] = F.pad(sl[i], (0, 0, 0, 0, 0, out2.shape[1]-sl[i].shape[1]))
        #         out2 = out2 + (1 / math.ceil(self.in_channels / self.out_channels)) * sl[i]
        out2 = self.bias12(out2)
        out2 = self.prelu2(out2)
        out2 = self.bias12(out2)

        out3 = out1 + out2

        return out3

class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

class LsqConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(LsqConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.bit = 4
        self.quan_w_fn = LsqQuan(bit=8)
        self.quan_a_fn = LsqQuan(all_positive=True, bit=8)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(self.weight.detach())
        self.quan_w_fn.init_from(self.weight)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return torch.nn.functional.conv2d(quantized_act, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
'''
