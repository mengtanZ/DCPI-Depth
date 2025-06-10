from __future__ import absolute_import, division, print_function

import time
import torch.optim as optim
from kornia.geometry import depth_to_normals
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from networks.layers import *

import datasets
import networks
from networks.RAFT.raft import RAFT
from networks.linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

from networks.normal_ranking_loss import EdgeguidedNormalRankingLoss

import os

import random
import numpy as np

from utils.utils import *

seed = 640
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:{}".format(self.opt.device_id))

        self.profile = self.opt.profile
        self.zero_tensor = (torch.tensor(0.0, device=self.device, requires_grad=False),
                            torch.tensor(0.0, device=self.device, requires_grad=False),
                            torch.tensor(0.0, device=self.device, requires_grad=False))

        self.num_scales = len(self.opt.scales)
        self.frame_ids = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ******************************* PREPARING MODELS FOR TRAINING **********************************************
        # prepareing DepthNet
        if self.opt.baseline == 'lite':
            self.models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                       drop_path_rate=self.opt.drop_path,
                                                       width=self.opt.width, height=self.opt.height)
            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                         self.opt.scales)

        elif self.opt.baseline == 'diff':
            self.models["encoder"] = networks.test_hr_encoder.hrnet18(True)
            self.models["encoder"].num_ch_enc = [64, 18, 36, 72, 144]
            self.models["depth"] = networks.HRDepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)

        elif self.opt.baseline == 'mono2':
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, pretrained=True)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # prepareing PoseNet
        self.models_pose["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)

        self.models_pose["pose"] = networks.PoseDecoder(
            self.models_pose["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)

        self.models_pose["pose_encoder"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())
        self.models_pose["pose"].to(self.device)
        self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        # prepareing FlowNet
        self.models["flow"] = torch.nn.DataParallel(RAFT(self.opt))
        self.models["flow"].to(self.device)
        self.parameters_to_train += list(self.models["flow"].parameters())
        self.models["flow"].load_state_dict(torch.load('weights/raft-kitti.pth'), strict=True)

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"
            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # ***************************** PREPARING OPTIMIZER FOR TRAINING ********************************************
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3],
                                                    weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer,
            T_0=int(self.opt.lr[2]),
            T_mul=1,
            eta_min=self.opt.lr[1],
            last_epoch=-1,
            max_lr=self.opt.lr[0],
            warmup_steps=0,
            gamma=0.9
        )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.opt.mypretrain is not None:
            self.load_pretrain()

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.train_posenet == True:
            for param in self.models["encoder"].parameters():
                param.requires_grad = False
            for param in self.models["depth"].parameters():
                param.requires_grad = False
            for param in self.models["flow"].parameters():
                param.requires_grad = False

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # ******************************* PREPARING DATA FOR TRAINING **********************************************
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "ddad": datasets.PreprocessedDataset,
                         "nuscenes": datasets.PreprocessedDataset,
                         "waymo": datasets.PreprocessedDataset,
                         "kitti_pp": datasets.PreprocessedDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, psedo_depth=self.opt.psedo_depth)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, psedo_depth=self.opt.psedo_depth)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        # ******************************* PREPARING TRAINING TOOLS **********************************************
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.pixel2flow = Pixel2Flow(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
        self.rectify_pixel = Rectify_Pixel(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
        self.backproject_depth_ps = BackprojectDepth_Ps(self.opt.batch_size, self.opt.height, self.opt.width).to(
            self.device)
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):

            # import ipdb
            # ipdb.set_trace()

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                # self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                # self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        outputs.update(self.predict_poses(inputs, features))

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        outputs["flow"] = self.models["flow"](inputs[("color", 0, 0)] * 255, inputs[("color", 1, 0)] * 255, iters=24,
                                              test_mode=True)[1]

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models_pose["pose"](pose_inputs)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                zero_translation = translation[:, 0] * torch.tensor([0, 0, 0], dtype=translation.dtype,
                                                                    device=translation.device)
                outputs[("cam_T_cam_purer", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], zero_translation, invert=(f_i < 0))

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                    R = outputs[("cam_T_cam_purer", 0, frame_id)]  # R

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                if frame_id == 1:
                    flow = outputs["flow"].clone().detach()  # 1

                    r_pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], R)

                    outputs[("rot_flow", frame_id)] = self.pixel2flow(r_pix_coords).detach()

                    rigid_pix_coords = pix_coords.clone()
                    outputs[("rig_flow", frame_id)] = self.pixel2flow(rigid_pix_coords).detach()

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).to(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss

            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        depth = outputs[("depth", 0, 0)]
        flow = outputs["flow"]

        if self.opt.version == 'v1':
            rot_flow = outputs[("rot_flow", 1)]
            rig_flow = outputs[("rig_flow", 1)]
            dcpi_loss = self.compute_dcpi_loss(depth, outputs[("cam_T_cam", 0, 1)], flow,
                                                                       rot_flow, rig_flow, inputs[("K", 0)])

        elif self.opt.version == 'v2':
            rot_flow = outputs[("rot_flow", 1)]
            rig_flow = outputs[("rig_flow", 1)]
            t = -outputs[("cam_T_cam", 0, 1)][:, 0:3, 3]
            dcpi_loss = self.compute_dcpi_loss_v2(depth, t, flow, rot_flow, rig_flow)

        total_loss += self.opt.dcpi_weights * dcpi_loss

        if self.epoch < 1:
            total_loss -= self.opt.dcpi_weights * dcpi_loss

        if self.opt.psedo_depth != "none":
            if not inputs["psedo_depth"].shape == outputs[("depth", 0, 0)].shape:
                inputs["psedo_depth"] = F.interpolate(
                    inputs["psedo_depth"], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            tgt_normal = depth_to_normals(outputs[("depth", 0, 0)], inputs[('K', 0)][:, :3, :3])
            tgt_pseudo_normal = depth_to_normals(inputs['psedo_depth'], inputs[('K', 0)][:, :3, :3])
            # nomal loss
            loss_normal = (tgt_normal - tgt_pseudo_normal).abs().mean()
            total_loss += self.opt.psedo_weights * loss_normal  # 0.1 for kitti and 0.01 for ddad
            # normal ranking loss
            if self.opt.normal_ranking_loss:  # for ddad exp
                normal_ranking_loss = EdgeguidedNormalRankingLoss().to(self.device)
                loss_normalrank = normal_ranking_loss(inputs['psedo_depth'], outputs[("depth", 0, 0)], tgt_normal,
                                                      tgt_pseudo_normal)
                total_loss += self.opt.psedo_weights * loss_normalrank

        losses["dcpi_loss"] = dcpi_loss
        losses["loss"] = total_loss

        return losses

    def compute_dcpi_loss(self, tgt_depth, pose_mat, flow, rot_flow, rig_flow, intrinsics):
        batch, _, height, width = tgt_depth.size()
        device = tgt_depth.device
        loss_1 = torch.zeros(1).to(device).mean()
        loss_2 = torch.zeros(1).to(device).mean()
        loss_3 = torch.zeros(1).to(device).mean()

        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        pix = torch.stack([x_coords, y_coords], axis=0).to(device)
        p_t = pix.unsqueeze(0).repeat(batch, 1, 1, 1).to(torch.float)
        p_s = p_t + flow

        intrinsics_inv = torch.inverse(intrinsics)
        pose_mat = torch.inverse(pose_mat)[:, :3, :]

        q_t = torch.cat(
            [intrinsics_inv[:, i, i].view(batch, 1, 1, 1) * p_t[:, i:i + 1, :, :] + intrinsics_inv[:, i, 2].view(batch,
                                                                                                                 1,
                                                                                                                 1, 1)
             for i in range(2)], dim=1)
        q_s = torch.cat(
            [intrinsics_inv[:, i, i].view(batch, 1, 1, 1) * p_s[:, i:i + 1, :, :] + intrinsics_inv[:, i, 2].view(batch,
                                                                                                                 1,
                                                                                                                 1, 1)
             for i in range(2)], dim=1)

        I = torch.cat([pose_mat[:, i, 0].view(batch, 1, 1, 1) * q_t[:, 0:1, :, :]
                       + pose_mat[:, i, 1].view(batch, 1, 1, 1) * q_t[:, 1:2, :, :]
                       + pose_mat[:, i, 2].view(batch, 1, 1, 1)
                       for i in range(3)], dim=1)  # b3hw
        t = pose_mat[:, :, 3]  # 4,3

        rep_depth = -(torch.sum(t[:, 0:2], dim=1, keepdim=True).view(batch, 1, 1, 1) - t[:, 2:].view(batch, 1, 1,
                                                                                                     1) * torch.sum(
            q_s[:, 0:2, :, :], dim=1, keepdim=True)) / (
                            I[:, 2:, :, :] * torch.sum(q_s[:, 0:2, :, :], dim=1, keepdim=True) - torch.sum(
                        I[:, 0:2, :, :], dim=1, keepdim=True) + 1e-5)

        t_flow = (flow - rot_flow).detach()

        loss_1 = torch.abs((rep_depth - tgt_depth) / (tgt_depth + 1e-5))
        loss_1 = loss_1[loss_1 < 2].mean()

        if self.epoch > (2 * self.opt.num_epochs / 3):
            div_flow = compute_div(t_flow)
            grad_depth = compute_grad(tgt_depth)

            corelated_div_flow = div_flow * tgt_depth / t - 4
            corelated_grad_depth = compute_inner_product(grad_depth, p_t) / tgt_depth

            loss_2 = torch.abs((corelated_div_flow - corelated_grad_depth) / (corelated_grad_depth + 1e-5))
            loss_2 = loss_2[loss_2 < 0.2].mean()

            loss_3 = torch.abs((flow - rig_flow) / (flow + 1e-5))
            loss_3 = loss_3[loss_3 < 0.2].mean()

        loss = loss_1 + 0.2 * loss_2 + 0.2 * loss_3

        return loss

    def compute_dcpi_loss_v2(self, tgt_depth, t, flow, rot_flow, rig_flow):

        device = tgt_depth.device
        loss_1 = torch.zeros(1).to(device).mean()
        loss_2 = torch.zeros(1).to(device).mean()
        loss_3 = torch.zeros(1).to(device).mean()

        trans_flow = flow - rot_flow.detach()
        B, _, height, width = tgt_depth.size()
        center_y = (height - 1) / 2
        center_x = (width - 1) / 2

        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        y_coords_centered = y_coords - center_y
        x_coords_centered = x_coords - center_x

        pix = torch.stack([x_coords_centered, y_coords_centered], axis=0)
        p = pix.unsqueeze(0).repeat(B, 1, 1, 1).to(self.device)

        inner_p_F = compute_inner_product(p, trans_flow)
        inner_p_p = compute_inner_product(p, p)

        t = t[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width).to(self.device)

        pro_D = t * (inner_p_p + inner_p_F) / (inner_p_F + 1e-5)

        loss_1 = torch.abs((pro_D - tgt_depth) / (tgt_depth + 1e-5))
        loss_1 = loss_1[loss_1 < 2].mean()

        loss_3 = torch.abs((flow - rig_flow) / (flow + 1e-5))
        loss_3 = loss_3[loss_3 < 0.5].mean()

        if self.epoch > (2 * self.opt.num_epochs / 3):
            div_flow = compute_div(trans_flow)
            grad_depth = compute_grad(tgt_depth)

            corelated_div_flow = div_flow * tgt_depth / t - 4
            corelated_grad_depth = compute_inner_product(grad_depth, p) / tgt_depth

            loss_2 = torch.abs((corelated_div_flow - corelated_grad_depth) / (corelated_grad_depth + 1e-5))
            loss_2 = loss_2[loss_2 < 0.2].mean()

        loss = loss_1 + 0.2 * loss_2 + 0.2 * loss_3

        return loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        # inputs["depth_gt"].reshape(inputs[("color", 0, 0)].shape)

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state

        # if self.opt.train_posenet == False:
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_pose_load_path):
            print("Loading Adam weights")
            # if self.opt.train_posenet == False:
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path, map_location=self.device)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
