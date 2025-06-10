import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

# def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
#     """
#     depth: (H, W)
#     """
#     x = depth.cpu().numpy()
#     x = np.nan_to_num(x)  # change nan to 0
#     mi = np.min(x)  # get minimum depth
#     ma = np.max(x)
#     x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
#     x = (255*x).astype(np.uint8)
#     x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
#     x_ = T.ToTensor()(x_)  # (3, H, W)
#     return x_

def visualize_depth(depth, cmap='magma'):
    """
    depth: (H, W)
    """
    x = depth.cpu().detach().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    x = 1 / (x + 1e-3)
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8) # normalize to 0~1
    # x = 1-x
    x = (255 * x).astype(np.uint8)

    # Apply 'magma' colormap
    colormap = plt.get_cmap(cmap)
    x_color = colormap(x)

    # Convert to PIL image
    x_pil = Image.fromarray((x_color * 255).astype(np.uint8))

    # Convert to PyTorch tensor
    x_tensor = T.ToTensor()(x_pil)  # (3, H, W)
    return x_tensor

def visualize_gt_depth(depth, cmap='magma'):
    """
    depth: (H, W)
    """
    x = depth.cpu().detach().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    # x = 1 / (x + 1e-3)
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8) # normalize to 0~1
    x = 1 / (x + 1e-3)
    # x = 1-x
    x = (255 * x).astype(np.uint8)

    # Apply 'magma' colormap
    colormap = plt.get_cmap(cmap)
    x_color = colormap(x)

    # Convert to PIL image
    x_pil = Image.fromarray((x_color * 255).astype(np.uint8))

    # Convert to PyTorch tensor
    x_tensor = T.ToTensor()(x_pil)  # (3, H, W)
    return x_tensor


def visualize_error(error, cmap='jet'):
    """
    Visualizes the error map using the 'jet' colormap.
    error: (H, W) error map with NaN values in some regions.
    """
    # Replace NaNs with zero for visualization
    x = np.nan_to_num(error, nan=0.0)

    # Set a threshold for better contrast (adjust if necessary)
    max_threshold = 0.5
    x[x > max_threshold] = max_threshold

    # Normalize to the range [0, 1] for the colormap
    x = x / max_threshold

    # Apply the 'jet' colormap
    colormap = plt.get_cmap(cmap)
    x_color = colormap(x)

    # Convert to PIL image
    x_pil = Image.fromarray((x_color[:, :, :3] * 255).astype(np.uint8))  # Only RGB channels

    # Convert to PyTorch tensor
    x_tensor = T.ToTensor()(x_pil)  # (3, H, W)
    return x_tensor


def visualize_image(image):
    """
    tensor image: (3, H, W)
    """
    x = (image.cpu() * 0.225 + 0.45)
    return x

def vis_tensor(img: torch.Tensor, name='default'):
    assert (len(img.shape) == 3 and img.shape[0] in [1, 3]) or len(img.shape) == 2
    if img.shape[0] == 1:
        img = img[0]
    vis = img.detach().cpu().numpy()
    if len(vis.shape) == 3:
        vis = np.transpose(vis, [1, 2, 0]) * 0.225 + 0.45
    plt.figure(name)
    plt.imshow(vis)
    plt.axis('off')

    save_path = f'./vis/{name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close()

def visualize_flow(flow):
    flo = flow.permute(1, 2, 0).cpu().detach().numpy()
    flo = flow_to_image(flo)
    return flo

def vis_flow(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().detach().numpy()

    # map flow to rgb image
    flo = flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)