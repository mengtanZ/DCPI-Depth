import torch
import numpy as np

def make_colorwheel(device='cpu'):
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Args:
        device (str): Device to place the color wheel tensor (default is 'cpu')

    Returns:
        torch.Tensor: Color wheel of shape [55, 3]
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3), device=device)
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY, device=device).float() / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG, device=device).float() / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = torch.floor(255 * torch.arange(0, GC, device=device).float() / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB, device=device).float() / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = torch.floor(255 * torch.arange(0, BM, device=device).float() / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR, device=device).float() / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (torch.Tensor): Input horizontal flow of shape [H,W]
        v (torch.Tensor): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        torch.Tensor: Flow visualization image of shape [H,W,3]
    """
    flow_image = torch.zeros((u.shape[0], u.shape[1], 3), dtype=torch.uint8, device=u.device)
    colorwheel = make_colorwheel(device=u.device)  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    a = torch.atan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).to(torch.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0.float()
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = torch.floor(255 * col)
    return flow_image


def batch_flow_to_images(batch_flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Convert a batch of flow images to RGB images.

    Args:
        batch_flow_uv (torch.Tensor): Batch of flow UV images of shape [B,H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        torch.Tensor: Batch of flow visualization images of shape [B,H,W,3]
    """
    assert batch_flow_uv.ndim == 4, 'input batch flow must have four dimensions'
    assert batch_flow_uv.shape[3] == 2, 'input batch flow must have shape [B,H,W,2]'
    batch_size = batch_flow_uv.shape[0]
    device = batch_flow_uv.device
    batch_images = []
    for i in range(batch_size):
        flow_image = flow_uv_to_colors(batch_flow_uv[i, :, :, 0], batch_flow_uv[i, :, :, 1], convert_to_bgr)
        batch_images.append(flow_image)
    return torch.stack(batch_images).to(device)
