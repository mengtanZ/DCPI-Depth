from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlockDepth(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()

        self.conv = DepthConv3x3(in_channels, out_channels)
        self.nonlin = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(DepthConv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=3 // 2, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class BackprojectDepth_Ps(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth_Ps, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, flow):
        # flow_new = flow.permute(0, 2, 3, 1)
        # flow_new[..., 0] /= self.width - 1
        # flow_new[..., 1] /= self.height - 1
        # flow_new = flow_new.permute(0, 3, 1, 2)
        # self.pix_coords = self.pix_coords.to(flow.device)
        # print(flow.device)
        # print(self.pix_coords.device)
        flow_expanded = torch.cat((flow, torch.zeros(flow.size(0), 1, flow.size(2), flow.size(3), device=flow.device)), dim=1)
        flow_flat = flow_expanded.view(flow_expanded.size(0), 3, -1)
        updated_pix_coords = self.pix_coords + flow_flat
        cam_points = torch.matmul(inv_K[:, :3, :3], updated_pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Pixel2Flow(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(Pixel2Flow, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        u_grid, v_grid = torch.meshgrid(torch.arange(self.height, requires_grad=False), torch.arange(self.width, requires_grad=False))
        self.pix_coords = torch.cat([v_grid.expand(self.batch_size, 1, self.height, self.width), u_grid.expand(self.batch_size, 1, self.height, self.width)], dim=1)

        # meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        # self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
        #                               requires_grad=False)
        # self.pix_coords = torch.unsqueeze(torch.stack(
        #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        # self.pix_coords = torch.unsqueeze(torch.stack(
        #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0).view(1, 2, self.height, self.width)

    def forward(self, new_pixel):
        self.pix_coords = self.pix_coords.to(new_pixel.device)
        new_pixel += 1
        new_pixel[..., 0] *= self.width / 2
        new_pixel[..., 1] *= self.height / 2
        new_pixel = new_pixel.permute(0, 3, 1, 2)

        # print(self.pix_coords.size())
        # print(new_pixel.size())
        flow = new_pixel - self.pix_coords

        return flow

class Rectify_Pixel(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Rectify_Pixel, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points_t, points_s, K, res1_T, res2_T):
        P_res1 = torch.matmul(K, res1_T)[:, :3, :]
        P_res2 = torch.matmul(K, res2_T)[:, :3, :]

        cam_points_res1 = torch.matmul(P_res1, points_t)
        cam_points_res2 = torch.matmul(P_res2, points_t)
        cam_points_s = torch.matmul(K[:, :3, :], points_s)
        cam_points_t = torch.matmul(K[:, :3, :], points_t)

        # print(cam_points_s.size())
        # print(cam_points_res1.size())

        rec1_cam_flow = cam_points_s - cam_points_res1
        # rec2_cam_flow = cam_points_res1 - cam_points_res2
        rec2_cam_flow = cam_points_s - cam_points_res2
        rec1_cam_points = cam_points_t + rec1_cam_flow # pure t3 contribute to, decomposed from R, t
        rec2_cam_points = cam_points_t + rec2_cam_flow # pure t1, t2 contribute to, decomposed from R, t1, t2

        rec1_pix_coords = rec1_cam_points[:, :2, :] / (rec1_cam_points[:, 2, :].unsqueeze(1) + self.eps)
        rec1_pix_coords = rec1_pix_coords.view(self.batch_size, 2, self.height, self.width)
        rec1_pix_coords = rec1_pix_coords.permute(0, 2, 3, 1)
        # rec1_pix_coords[..., 0] /= self.width - 1
        # rec1_pix_coords[..., 1] /= self.height - 1
        # rec1_pix_coords = (rec1_pix_coords - 0.5) * 2

        rec2_pix_coords = rec2_cam_points[:, :2, :] / (rec2_cam_points[:, 2, :].unsqueeze(1) + self.eps)
        rec2_pix_coords = rec2_pix_coords.view(self.batch_size, 2, self.height, self.width)
        rec2_pix_coords = rec2_pix_coords.permute(0, 2, 3, 1)
        # rec2_pix_coords[..., 0] /= self.width - 1
        # rec2_pix_coords[..., 1] /= self.height - 1
        # rec2_pix_coords = (rec2_pix_coords - 0.5) * 2

        pix_coords_s = cam_points_s[:, :2, :] / (cam_points_s[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords_s = pix_coords_s.view(self.batch_size, 2, self.height, self.width)
        pix_coords_s = pix_coords_s.permute(0, 2, 3, 1)
        pix_coords_s[..., 0] /= self.width - 1
        pix_coords_s[..., 1] /= self.height - 1
        pix_coords_s = (pix_coords_s - 0.5) * 2

        return rec1_pix_coords, rec2_pix_coords, pix_coords_s


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_div(flow):
    _, _, h, w = flow.shape
    flow_padded = F.pad(flow, (1, 1, 1, 1), mode='constant', value=0)

    # div_flow_x = - flow_padded[:, 1:, :h, 1:w + 1] \
    #              + flow_padded[:, 1:, 2:, 1:w + 1]
    # div_flow_y = + flow_padded[:, 0:1, 1:h + 1, 2:] \
    #              - flow_padded[:, 0:1, 1:h + 1, :w]

    div_flow = - flow_padded[:, 1:, :h, 1:w + 1] \
               + flow_padded[:, 1:, 2:, 1:w + 1] \
               + flow_padded[:, 0:1, 1:h + 1, 2:] \
               - flow_padded[:, 0:1, 1:h + 1, :w]
    return div_flow


def compute_cur(flow):
    _, _, h, w = flow.shape
    flow_padded = F.pad(flow, (1, 1, 1, 1), mode='constant', value=0)

    cur_flow = - flow_padded[:, 0:1, :h, 1:w + 1] \
               + flow_padded[:, 0:1, 2:, 1:w + 1] \
               - flow_padded[:, 1:, 1:h + 1, 2:] \
               + flow_padded[:, 1:, 1:h + 1, :w]
    return cur_flow


def compute_grad(depth):
    _, _, h, w = depth.shape
    flow_padded = F.pad(depth, (1, 1, 1, 1), mode='constant', value=0)

    grade_depth_x = flow_padded[:, :, 2:, 1:w + 1] - flow_padded[:, :, :h, 1:w + 1]
    grade_depth_y = flow_padded[:, :, 1:h + 1, 2:] - flow_padded[:, :, 1:h + 1, :w]

    # grade_depth_x = -flow_padded[:, :, 2:, 1:w + 1] + flow_padded[:, :, :h, 1:w + 1]
    # grade_depth_y = -flow_padded[:, :, 1:h + 1, 2:] + flow_padded[:, :, 1:h + 1, :w]

    grade_depth = torch.cat([grade_depth_y, grade_depth_x], dim=1)

    return grade_depth


def compute_inner_product(a, b):
    _, _, h, w = a.shape
    inner = torch.sum(a[:, 0:1, :, :] * b[:, 0:1, :, :] + a[:, 1:2, :, :] * b[:, 1:2, :, :], dim=1, keepdim=True)
    return inner


def compute_unit_vector(a):
    magnitude = torch.sqrt(a[:, 0:1, :, :] ** 2 + a[:, 1:2, :, :] ** 2) + 1e-5
    normalized_a = a / magnitude
    return normalized_a


def normalization_tensor(tensor):
    min_val = torch.min(torch.min(tensor, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
    max_val = torch.max(torch.max(tensor, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]

    # 对 b1hw 张量在后两维的切片上进行归一化
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor




