from __future__ import absolute_import, division, print_function

import csv
import os

from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from networks.layers import disp_to_depth
from utils.utils import readlines
from options import Options
import datasets
import networks
import time
from thop import clever_format
from thop import profile

from visialization import visualize_depth
from imageio import imwrite

from scipy import sparse

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e,), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d,), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = np.array(sparse_depth.todense(), dtype=np.float16)
    return depth


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    if opt.dataset == "ddad":
        MAX_DEPTH = 200
    elif opt.dataset in ["nuscenes", "waymo"]:
        MAX_DEPTH = 75

    device = torch.device("cpu" if opt.no_cuda else "cuda:{}".format(opt.device_id))

    # ******************************* LOADING PRETRAINED WEIGHTS **********************************************
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    # ******************************* PREPARING MODELS FOR EVALUATION **********************************************
    if opt.baseline == 'lite':
        encoder = networks.LiteMono(model=opt.model,
                                    height=encoder_dict['height'],
                                    width=encoder_dict['width'])
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=opt.scales)
    elif opt.baseline == 'diff':
        encoder = networks.test_hr_encoder.hrnet18(False)
        encoder.num_ch_enc = [64, 18, 36, 72, 144]
        depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, opt.scales)
    elif opt.baseline == 'mono2':
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder_mono2(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    depth_model_dict = depth_decoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict}, strict=False)
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict},
                                  strict=False)
    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    # ******************************* PREPARING DATA FOR EVALUATION **********************************************
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    print("Evaluation on the {} datasets with {} images".format(opt.dataset, len(filenames)))
    if opt.dataset == 'kitti':
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 1, is_train=False)
    elif opt.dataset == 'make3d':
        dataset = datasets.Make3DDataset(opt.data_path, filenames,
                                         (encoder_dict['height'], encoder_dict['width']))
    elif opt.dataset in ['nuscenes', 'waymo', 'ddad', 'diml']:
        dataset = datasets.PreprocessedDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               [0], 1, is_train=False, img_ext='.jpg')

    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    # ******************************* LOADING GT FOR EVALUATION **********************************************
    if opt.eval_split == "ddad":
        # gt_path = '/remote-home/mtzhang/dataset/ddad/testing/depth'
        gt_path = 'D:/data/ddad/testing/depth'
        gt_depths = sorted([file for file in os.listdir(gt_path) if file.endswith(".npz")])
        gt_depths = [load_sparse_depth("{}/{}".format(gt_path, gt_depth)) for gt_depth in gt_depths]
    elif opt.eval_split == "nuscenes":
        # gt_path = '/remote-home/mtzhang/dataset/nuscenes_preprocessed/testing/depth'
        gt_path = 'D:/data/nuscenes/testing/depth'
        gt_depths = [np.load(f'{gt_path}/{file}') for file in sorted(os.listdir(gt_path)) if file.endswith('.npy')]
    elif opt.eval_split == "cityscapes":
        gt_path = os.path.join(os.path.dirname(__file__), "splits", "cityscapes", "gt_depths")
        gt_depths = [np.load(f'{gt_path}/{file}') for file in sorted(os.listdir(gt_path)) if file.endswith('.npy')]
    elif opt.eval_split == "eigen_benchmark":
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    elif opt.eval_split == "make3d":
        gt_path = '/remote-home/mtzhang/dataset/make3d/Test134'
        gt_depths = [np.load(f'{gt_path}/{file}') for file in sorted(os.listdir(gt_path)) if file.endswith('.npy')]
    else:
        # gt_path = '/remote-home/mtzhang/dataset/kitti/testing/depth'
        gt_path = 'D:/data/kitti/testing/depth'
        gt_depths = [np.load(f'{gt_path}/{file}') for file in sorted(os.listdir(gt_path)) if file.endswith('.npy')]

    # ******************************* INFERENCE AND SAVE RESULTS **********************************************
    pred_disps = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    idx = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].to(device)

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
            output = depth_decoder(encoder(input_color))

            pred_disp, pred_depth_vis = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            if opt.save_vis:
                vis = visualize_depth(pred_depth_vis[0, 0]).permute(
                    1, 2, 0).numpy() * 255
                imwrite('pred_depth/{}_{}/{}.jpg'.format(opt.eval_split, opt.model_name, idx),
                        vis.astype(np.uint8))

            pred_disp = pred_disp.cpu()[:, 0].numpy()
            idx += 1

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # ******************************* EVALUATION RESULTS WITH GT **********************************************

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        # get gt depth. For nuscenes and waymo, it is saved as point cloud
        gt_depth = gt_depths[i]
        # batch_size = 1 # must be 1

        # get corresponding predicted depth and resize to gt resolution
        pred_disp = pred_disps[i]
        if opt.eval_split in ["eigen", "ddad"]:
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        else:
            pred_disp = cv2.resize(pred_disp, (opt.gt_width, opt.gt_height))
        pred_depth = 1 / pred_disp

        # make valid mask to align pred depth with gt depth for different datasets
        if opt.eval_split in ["nuscenes", "waymo"]:

            x_pixel = gt_depth[:, 0]
            y_pixel = gt_depth[:, 1]
            depth_values = gt_depth[:, 2]

            # check the point cloud in image
            valid_mask = (
                    (x_pixel >= 0) & (x_pixel < opt.gt_width) &
                    (y_pixel >= 0) & (y_pixel < opt.gt_height) &
                    (depth_values > MIN_DEPTH) & (depth_values < MAX_DEPTH)  # 检查深度值范围
            )
            # get the valid depth array and the valid pixels in image
            gt_depth = depth_values[valid_mask]  # shape (xxx',)
            valid_x = x_pixel[valid_mask].astype(int)
            valid_y = y_pixel[valid_mask].astype(int)

            # align the pred depth with valid gt depth array
            pred_depth = pred_depth[valid_y, valid_x]  # shape (xxx',)

        elif opt.eval_split in ["eigen", "ddad"]:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)  # 375x1242

            if opt.eval_split == "eigen":
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1  # 375x1242
                mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

        elif opt.eval_split == "diml":
            pred_depth[pred_depth > 10] = 10

        else:
            pred_depth[pred_depth < opt.min_depth] = opt.min_depth
            pred_depth[pred_depth > opt.max_depth] = opt.max_depth

            pred_depth *= opt.pred_depth_scale_factor

        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        errors.append(compute_errors(gt_depth, pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops,
                                                                                                             params,
                                                                                                             flops_e,
                                                                                                             params_e,
                                                                                                             flops_d,
                                                                                                             params_d))

    print("\n-> Done!")

    # ******************************* SAVE THE EVALUATION RESULT **********************************************
    header = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    data = mean_errors.tolist()
    data.append(opt.load_weights_folder.split("\\")[-1])

    csv_filename = "eval/{}.csv".format(opt.model_name)
    file_exist = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        if not file_exist:
            csv_writer.writerow(header)

        csv_writer.writerow(data)


if __name__ == "__main__":
    options = Options()
    evaluate(options.parse())
