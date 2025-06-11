from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="DCPI-Depth options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="tmp")
        self.parser.add_argument("--device_id",
                                 type=int,
                                 help="gpu number",
                                 default=0)
        self.parser.add_argument("--vis_device_id",
                                 type=int,
                                 help="visible gpu number",
                                 default=0)

        # TRAINING options
        self.parser.add_argument("--baseline",
                                 type=str,
                                 choices=["lite", "diff", "mono2"],
                                 help="the name of the folder to save the model in",
                                 default="lite")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="DCPI-Depth")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_zhou_pp", "ddad", "nuscenes", "waymo"],
                                 default="eigen_zhou")
        self.parser.add_argument("--model",
                                 type=str,
                                 help="which model to load",
                                 choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"],
                                 default="lite-mono")
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay in AdamW",
                                 default=1e-2)
        self.parser.add_argument("--drop_path",
                                 type=float,
                                 help="drop path rate",
                                 default=0.2)
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_pp", "ddad", "nuscenes", "waymo", "make3d", "diml"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=384)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set, uses improved gt",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        self.parser.add_argument("--profile",
                                 type=bool,
                                 help="profile once at the beginning of the training",
                                 default=True)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        self.parser.add_argument("--lr",
                                 nargs="+",
                                 type=float,
                                 help="learning rates of DepthNet and PoseNet. "
                                      "Initial learning rate, "
                                      "minimum learning rate, "
                                      "First cycle step size.",
                                 default=[0.0001, 5e-6, 31, 0.0001, 1e-5, 31])
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=50)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--mypretrain",
                                 type=str,
                                 help="if set, use my pretrained encoder")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--gt_height",
                                 type=int,
                                 help="input image height",
                                 default=900)
        self.parser.add_argument("--gt_width",
                                 type=int,
                                 help="input image width",
                                 default=1600)
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                     "eigen", "eigen_benchmark", "ddad", "nuscenes", "waymo", "make3d", "diml"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--save_vis",
                                 help="if set, save depth predictions as .png",
                                 action="store_true")

        self.parser.add_argument('--small', action='store_true',
                                 help='use small model')
        self.parser.add_argument('--mixed_precision', action='store_true',
                                 help='use mixed precision')
        self.parser.add_argument('--iters', type=int, default=1)

        self.parser.add_argument("--flow_choose",
                                 type=str,
                                 default="refined_255",
                                 choices=["refined_255", "refined_1", "coarse_255", "coarse_1"])
        self.parser.add_argument("--version",
                                 type=str,
                                 default="v2",
                                 choices=["v1", "v2", "test"])
        self.parser.add_argument("--dcpi_weights",
                                 type=float,
                                 default=0.01)
        self.parser.add_argument("--weights",
                                 type=float,
                                 nargs='+',
                                 default=[1.0, 0.25, 0.1, 0.05],
                                 help="A list of weights")
        self.parser.add_argument('--staged', action='store_true',
                                 help='use staged optimization')
        self.parser.add_argument("--separate",
                                 type=str,
                                 default="w2",
                                 choices=["w2", "wo2"])
        self.parser.add_argument("--share",
                                 type=str,
                                 default="and",
                                 choices=["and", "or", "no"])
        self.parser.add_argument("--train_posenet",
                                 type=bool,
                                 default=False)

        # scipad
        self.parser.add_argument("--embed_dim",
                                 type=int,
                                 default=96)
        self.parser.add_argument("--backbone",
                                 type=str,
                                 default='resnet18')

        self.parser.add_argument("--psedo_depth",
                                 type=str,
                                 default="none",
                                 choices=["none", "leres_depth", "psedo_depth"])
        self.parser.add_argument("--normal_ranking_loss",
                                 action="store_true")
        self.parser.add_argument("--psedo_weights",
                                 type=float,
                                 default=0.01)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
