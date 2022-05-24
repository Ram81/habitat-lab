#!/bin/bash
source ./sbatch_scripts/training_function.sh

set -x

## Slurm 
REPO_NAME="objectnav"
MAIN_USER="alexclegg"
REPO_PATH="/private/home/${MAIN_USER}/karmesh_code/${REPO_NAME}/habitat-lab"
PARTITION="learnfair,learnlab,devlab"
SPLIT="train"
VAL_SPLIT="val"
BASE_TASK_CONFIG_PATH="${REPO_PATH}/configs/tasks/objectnav_mp3d_il.yaml"
EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_rgbd_objectnav.yaml"
NODES=8
#WANDB_MODE="online"
ENVIRONMENT="objectnav_mp3d_40k"
VIDEO_OPTION="[]"
#NUM_STEPS=5e8
NUM_UPDATES=16000
TIME="72:00:00"
#NUM_ENV=10
TEST_EPISODE_COUNT=10000
NUM_PROCESSES=8
RUN_TRAIN_SCRIPT=false
RUN_EVAL_SCRIPT=true

#EXP_NAME="objectnav_pretrained"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#	    IL.BehaviorCloning.num_steps 64 \
#	    IL.BehaviorCloning.num_mini_batch 2"
#SPLIT="train"
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_pretrained_nccl"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#	    IL.distrib_backend NCCL"
#SPLIT="train"
#CHKP_NAME="17"
#run_training 0

#EXP_NAME="objectnav_pretrained_augmentation"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#	    TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 640 \
#	    TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 480 \
#	    TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 640 \
#	    TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 480"
#SPLIT="train"
#CHKP_NAME="158"
#run_training 0

#EXP_NAME="objectnav_pretrained_augmentation_imagenav_augs"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2"
#SPLIT="train"
#CHKP_NAME="89"
#run_training 0
#CHKP_NAME="99"
#run_training 0
#CHKP_NAME="109"
#run_training 0
#CHKP_NAME="119"
#run_training 0
#CHKP_NAME="129"
#run_training 0
#CHKP_NAME="139"
#run_training 0
#CHKP_NAME="149"
#run_training 0
#CHKP_NAME="159"
#run_training 0


#EXP_NAME="objectnav_pretrained_augmentation_imagenav_augs_color_0_2"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.2 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.2 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.2 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.2 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2"
#SPLIT="train"
#CHKP_NAME="89"
#run_training 0
#CHKP_NAME="99"
#run_training 0
#CHKP_NAME="109"
#run_training 0
#CHKP_NAME="119"
#run_training 0
#CHKP_NAME="129"
#run_training 0
#CHKP_NAME="139"
#run_training 0
#CHKP_NAME="149"
#run_training 0
#CHKP_NAME="159"
#run_training 0

#EXP_NAME="objectnav_pretrained_augmentation_imagenav_translate_4"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2"
#SPLIT="train"
#CHKP_NAME="89"
#run_training 0
#CHKP_NAME="99"
#run_training 0
#CHKP_NAME="109"
#run_training 0
#CHKP_NAME="119"
#run_training 0
#CHKP_NAME="129"
#run_training 0
#CHKP_NAME="139"
#run_training 0
#CHKP_NAME="149"
#run_training 0
#CHKP_NAME="159"
#run_training 0

#EXP_NAME="objectnav_pretrained_augmentation_imagenav_augs_color_jitter_0_4"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.4 \
#	    IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2"
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_pretrained_augmentation_imagenav_augs_translate_32"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 32 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2"
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_pretrained_augmentation_imagenav_augs_translate_32_cj_0_4"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 32 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2"
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_scratch_rgbd"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS - \
#            MODEL.RGB_ENCODER.pretrained_ckpt \"None\""
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_scratch_augmentation_imagenav_augs"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2 \
# 	     MODEL.RGB_ENCODER.pretrained_ckpt \"None\""
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_scratch_augmentation_imagenav_augs_color_jitter_0_4"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 64 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.4 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2 \
#            MODEL.RGB_ENCODER.pretrained_ckpt \"None\""
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_pretrained_semantics"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 32 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL"
#EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_sem_seg_objectnav.yaml"
#NODES=16
#SPLIT="train"
#CHKP_NAME="15"
#run_training 0
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="17"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="19"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_scratch_semantics"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 32 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2 \
#            MODEL.RGB_ENCODER.pretrained_ckpt \"None\""
#EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_sem_seg_objectnav.yaml"
#NODES=16
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_pretrained_semantics_imagenav_augs"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 32 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v2"
#EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_sem_seg_objectnav.yaml"
#NODES=16
#SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0

#EXP_NAME="objectnav_pretrained_semantics_fullimage"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 32 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS - \
#	    IL.OBS_TRANSFORMS.ENABLED_TRANSFORMS () \
#	    TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 640 \
#	    TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 480 \
#            TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH 640 \
#            TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT 480 \
#            TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 640 \
#            TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 480"
#EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_sem_seg_objectnav.yaml"
#NODES=16
#SPLIT="train"
#CHKP_NAME="14"
#run_training 0
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0


#EXP_NAME="objectnav_pretrained_semantics_fullimage_augmentation"
#WEIGHTS_NAME="omnidata_DINO_02.pth"
#BACKBONE="resnet50_gn"
#EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
#            IL.BehaviorCloning.num_steps 32 \
#            IL.BehaviorCloning.num_mini_batch 2 \
#            IL.distrib_backend NCCL \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
#            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
#            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
#            IL.OBS_AUGMENTATIONS color_jitter-translate_v3 \
#            IL.OBS_TRANSFORMS.ENABLED_TRANSFORMS () \
#            TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 640 \
#            TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 480 \
#            TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH 640 \
#            TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT 480 \
#            TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 640 \
#            TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 480"
#EXP_CONFIG_PATH="${REPO_PATH}/habitat_baselines/config/objectnav/ddpil_ssl_sem_seg_objectnav.yaml"
#NODES=16
#SPLIT="train"
#CHKP_NAME="10"
#run_training 0
#CHKP_NAME="11"
#run_training 0

EXP_NAME="objectnav_pretrained_rgb_augmentations"
WEIGHTS_NAME="omnidata_DINO_02.pth"
BACKBONE="resnet50_gn"
EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
            IL.BehaviorCloning.num_steps 64 \
            IL.BehaviorCloning.num_mini_batch 2 \
            IL.distrib_backend NCCL \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
            IL.OBS_AUGMENTATIONS color_jitter-translate_v2 \
	    MODEL.DEPTH_ENCODER.cnn_type \"None\""
SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#Run_training 0
#CHKP_NAME="30"
#Run_training 0
#CHKP_NAME="31"
#Run_training 0
#CHKP_NAME="80"
#run_training 0
#CHKP_NAME="90"
#run_training 0
#CHKP_NAME="100"
#run_training 0
#CHKP_NAME="110"
#run_training 0
#CHKP_NAME="120"
#run_training 0
#CHKP_NAME="130"
#run_training 0
CHKP_NAME="140"
run_training 0
CHKP_NAME="150"
run_training 0
CHKP_NAME="155"
run_training 0


EXP_NAME="objectnav_scratch_rgb_augmentations"
WEIGHTS_NAME="omnidata_DINO_02.pth"
BACKBONE="resnet50_gn"
EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
            IL.BehaviorCloning.num_steps 64 \
            IL.BehaviorCloning.num_mini_batch 2 \
            IL.distrib_backend NCCL \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
            IL.OBS_AUGMENTATIONS color_jitter-translate_v2 \
            MODEL.RGB_ENCODER.pretrained_ckpt \"None\" \
	    MODEL.DEPTH_ENCODER.cnn_type \"None\""
SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0
#CHKP_NAME="80"
#run_training 0
#CHKP_NAME="90"
#run_training 0
#CHKP_NAME="100"
#run_training 0
#CHKP_NAME="110"
#run_training 0
#CHKP_NAME="120"
#run_training 0
#CHKP_NAME="130"
#run_training 0
CHKP_NAME="140"
run_training 0
CHKP_NAME="150"
run_training 0
CHKP_NAME="155"
run_training 0

EXP_NAME="objectnav_pretrained_rgb_augmentations_64_nopose"
WEIGHTS_NAME="omnidata_DINO_full_resnet_01.pth"
BACKBONE="resnet50_gn"
EXTRA_CMDS="TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.501094552723746 \
            IL.BehaviorCloning.num_steps 64 \
            IL.BehaviorCloning.num_mini_batch 2 \
            IL.distrib_backend NCCL \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.4 \
            IL.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 16 \
            IL.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 256 \
            IL.OBS_AUGMENTATIONS color_jitter-translate_v2 \
            MODEL.resnet_baseplanes 64 \
            MODEL.DEPTH_ENCODER.cnn_type \"None\""
SPLIT="train"
#CHKP_NAME="16"
#run_training 0
#CHKP_NAME="18"
#run_training 0
#CHKP_NAME="20"
#run_training 0
#CHKP_NAME="22"
#run_training 0
#CHKP_NAME="24"
#run_training 0
#CHKP_NAME="26"
#run_training 0
#CHKP_NAME="28"
#run_training 0
#CHKP_NAME="30"
#run_training 0
#CHKP_NAME="31"
#run_training 0
#CHKP_NAME="80"
#run_training 0
#CHKP_NAME="90"
#run_training 0
#CHKP_NAME="100"
#run_training 0
#CHKP_NAME="110"
#run_training 0
#CHKP_NAME="120"
#run_training 0
#CHKP_NAME="130"
#run_training 0
CHKP_NAME="140"
run_training 0
CHKP_NAME="150"
run_training 0
CHKP_NAME="155"
run_training 0
