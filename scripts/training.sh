export CUDA_VISIBLE_DEVICES=0

PRETRAIN=networks/lite-mono-pretrain.pth
PRETRAIN_8M=networks/lite-mono-8m-pretrain.pth

DATA_PATH=/remote-home/mtzhang/dataset/nuscenes_preprocessed/training
python train.py --dataset nuscenes --split nuscenes --data_path $DATA_PATH --model_name dcpi-nuscenes --model lite-mono --mypretrain $PRETRAIN --version v2 --num_epochs 10 --batch_size 6

DATA_PATH=/remote-home/mtzhang/dataset/ddad/training
python train.py --dataset ddad --split ddad --data_path $DATA_PATH --model_name dcpi-ddad --model lite-mono --mypretrain $PRETRAIN --version v2 --num_epochs 90 --batch_size 4 --psedo_depth leres_depth

DATA_PATH=/remote-home/mtzhang/dataset/kitti/training
python train.py --dataset kitti_pp --split eigen_zhou_pp --data_path $DATA_PATH --model_name dcpi-kitti --model lite-mono-8m --mypretrain $PRETRAIN_8M --version v2 --num_epochs 30 --batch_size 8 --width 640  --height 192 --psedo_depth leres_depth

DATA_PATH=/remote-home/mtzhang/dataset/waymo_preprocessed/training
python train.py --dataset waymo --split waymo --data_path $DATA_PATH --model_name dcpi-waymo --model lite-mono --mypretrain $PRETRAIN --version v2 --num_epochs 10 --batch_size 6