import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

import random

# generate zoom_image like Saliency-Sampler(ECCV2018)
def batch_augment(images, feature_map, mode='zoom'):
    batches, _, imgH, imgW = images.size()
    if mode == 'zoom':
        attention = torch.sum(feature_map.detach(), dim=1, keepdim=True)
        attention_map = nn.functional.interpolate(attention, size=(224, 224), mode='bilinear', align_corners=True)
        zoom_radius = ScaleLayer(0.08)
        grid_size = 31
        padding_size = 30
        global_size = grid_size + 2 * padding_size
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * padding_size + 1, fwhm=13))
        filter = nn.Conv2d(1, 1, kernel_size=(2 * padding_size + 1, 2 * padding_size + 1), bias=False)
        filter.weight[0].data[:, :, :] = gaussian_weights
        filter = filter.cuda()
        P_basis = torch.zeros(2, grid_size + 2 * padding_size, grid_size + 2 * padding_size)
        for kk in range(2):
            for ki in range(global_size):
                for kj in range(global_size):
                    P_basis[kk, ki, kj] = kk * (ki - padding_size) / (grid_size - 1.0) + (1.0 - kk) * (
                            kj - padding_size) / (grid_size - 1.0)
        P_basis = P_basis.cuda()

        xs = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            select_map = F.interpolate(atten_map, size=grid_size, mode='bilinear',
                                       align_corners=True)
            select_map_max = torch.max(select_map)
            decide_map = select_map / select_map_max
            zoom_p = random.sample(range(0, 3), 1)[0]
            temp = attention_zoom(decide_map=decide_map, radius=zoom_radius, grid_size=grid_size, p=zoom_p)
            xs.append(temp)

        xs = torch.cat(xs, 0)
        xs_hm = nn.ReplicationPad2d(padding_size)(xs)
        grid = create_grid(x=xs_hm, grid_size=grid_size, padding_size=padding_size, P_basis=P_basis,
                           global_size=global_size, filter=filter, input_size_net=imgH).to(images.device)
        x_sampled_zoom = F.grid_sample(images, grid)
        zoom_images = x_sampled_zoom

        return zoom_images
    else:
        raise ValueError(
            'Expected mode in [\'zoom\'], but received unsupported augmentation method %s' % mode)



def create_grid(x, grid_size, padding_size, P_basis, global_size, input_size_net, filter):
    P = torch.autograd.Variable(
        torch.zeros(1, 2, grid_size + 2 * padding_size, grid_size + 2 * padding_size).cuda(),
        requires_grad=False)
    P[0, :, :, :] = P_basis
    P = P.expand(x.size(0), 2, grid_size + 2 * padding_size, grid_size + 2 * padding_size)

    x_cat = torch.cat((x, x), 1)
    p_filter = filter(x)
    x_mul = torch.mul(P, x_cat).view(-1, 1, global_size, global_size)
    all_filter = filter(x_mul).view(-1, 2, grid_size, grid_size)

    x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, grid_size, grid_size)
    y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, grid_size, grid_size)

    x_filter = x_filter / p_filter
    y_filter = y_filter / p_filter

    xgrids = x_filter * 2 - 1
    ygrids = y_filter * 2 - 1
    xgrids = torch.clamp(xgrids, min=-1, max=1)
    ygrids = torch.clamp(ygrids, min=-1, max=1)

    xgrids = xgrids.view(-1, 1, grid_size, grid_size)
    ygrids = ygrids.view(-1, 1, grid_size, grid_size)

    grid = torch.cat((xgrids, ygrids), 1)

    grid = F.interpolate(grid, size=(input_size_net, input_size_net), mode='bilinear', align_corners=True)

    grid = torch.transpose(grid, 1, 2)
    grid = torch.transpose(grid, 2, 3)

    return grid

def makeGaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


class KernelGenerator(nn.Module):
    def __init__(self, size, offset=None):
        super(KernelGenerator, self).__init__()

        self.size = self._pair(size)
        xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
        if offset is None:
            offset_x = offset_y = size // 2
        else:
            offset_x, offset_y = self._pair(offset)
        self.factor = torch.from_numpy(-(np.power(xx - offset_x, 2) + np.power(yy - offset_y, 2)) / 2).float()

    @staticmethod
    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    def forward(self, theta):
        pow2 = torch.pow(theta * self.size[0], 2)
        kernel = 1.0 / (2 * np.pi * pow2) * torch.exp(self.factor.to(theta.device) / pow2)
        return kernel / kernel.max()


def kernel_generate(theta, size, offset=None):
    return KernelGenerator(size, offset)(theta)


def _mean_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)

class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices = F.max_pool2d(
            padded_maps,
            kernel_size=win_size,
            stride=1,
            return_indices=True)
        peak_map = (indices == element_map)

        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)

        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                   peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1) / \
                     (peak_map.view(batch_size, num_channels, -1).sum(2).view(batch_size, num_channels, 1, 1) + 1e-6)
        return (grad_input,) + (None,) * ctx.num_flags

def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)

class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value])).cuda()

    def forward(self, input):
        return input * self.scale

def attention_zoom(decide_map, radius=ScaleLayer(0.08), grid_size=31, base_ratio=0.09, p=0):

    H = decide_map.size(2)

    peak_list, aggregation = peak_stimulation(decide_map, win_size=3, peak_filter=_mean_filter)

    decide_map = decide_map.squeeze(0).squeeze(0)

    score = [decide_map[item[2], item[3]] for item in peak_list]
    x = [item[3] for item in peak_list]
    y = [item[2] for item in peak_list]

    if score == []:
        temp = torch.zeros(1, 1, grid_size, grid_size).cuda()
        temp += base_ratio
        xs = temp
        #xs_soft.append(temp)
        return xs

    peak_num = torch.arange(len(score))

    temp = base_ratio

    if p == 0:
        for i in peak_num:
            temp += score[i] * kernel_generate(radius(torch.sqrt(score[i])), H,
                                               (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).cuda()
    elif p == 1:
        for i in peak_num:
            rd = random.uniform(0, 1)
            if score[i] > rd:
                temp += score[i] * kernel_generate(radius(torch.sqrt(score[i])), H,
                                                   (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).cuda()
    elif p == 2:
        index = score.index(max(score))
        temp += score[index] * kernel_generate(radius(score[index]), H,
                                               (x[index].item(), y[index].item())).unsqueeze(0).unsqueeze(
            0).cuda()

    if type(temp) == float:
        temp += torch.zeros(1, 1, grid_size, grid_size).cuda()
    xs = temp

    return xs

