export CUDA_VISIBLE_DEVICES=0

WEIGHTS_PATH=/remote-home/mtzhang/mde-ssh/dcpi-depth/tmp

MODEL_NAME=dcpi-kitti
WEIGHTS=$WEIGHTS_PATH/$MODEL_NAME
DATA_PATH=/remote-home/share/KITTI/KITTI_fengyi
python evaluate_depth.py --dataset kitti --eval_split eigen --data_path $DATA_PATH --model lite-mono-8m --load_weights_folder $WEIGHTS --model_name $MODEL_NAME
#&   0.097  &   0.666  &   4.388  &   0.173  &   0.898  &   0.966  &   0.985  \\

MODEL_NAME=dcpi-ddad/models/weights_89
WEIGHTS=$WEIGHTS_PATH/$MODEL_NAME
DATA_PATH=/remote-home/mtzhang/dataset/ddad/testing
python evaluate_depth.py --dataset ddad --eval_split ddad --data_path $DATA_PATH --model lite-mono --load_weights_folder $WEIGHTS --model_name $MODEL_NAME
#&   0.141  &   2.711  &  14.757  &   0.236  &   0.813  &   0.931  &   0.971  \\

MODEL_NAME=dcpi-nuscenes
WEIGHTS=$WEIGHTS_PATH/$MODEL_NAME
DATA_PATH=/remote-home/mtzhang/dataset/nuscenes_preprocessed/testing
python evaluate_depth.py --dataset nuscenes --eval_split nuscenes --data_path $DATA_PATH --model lite-mono --load_weights_folder $WEIGHTS --model_name $MODEL_NAME --gt_height 900 --gt_width 1600
#&   0.157  &   1.795  &   7.192  &   0.255  &   0.790  &   0.914  &   0.959  \\

MODEL_NAME=dcpi-waymo
WEIGHTS=$WEIGHTS_PATH/$MODEL_NAME
DATA_PATH=/remote-home/mtzhang/dataset/waymo_preprocessed/testing
python evaluate_depth.py --dataset waymo --eval_split waymo --data_path $DATA_PATH --model lite-mono --load_weights_folder $WEIGHTS --model_name $MODEL_NAME --gt_height 1280 --gt_width 1920