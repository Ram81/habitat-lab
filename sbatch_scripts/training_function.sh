run_training() {
	     SEED=$1
	     # create run folder
	     RUN_FOLDER="/checkpoint/${USER}/${REPO_NAME}/${EXP_NAME}/${SEED}"
	     LOG_DIR="${RUN_FOLDER}/logs"
	     CHKP_DIR="${RUN_FOLDER}/chkp"
	     VIDEO_DIR="${RUN_FOLDER}/videos"
	     RESULTS_DIR="${RUN_FOLDER}/results"
	     CMD_TRAIN_OPTS_FILE="${LOG_DIR}/cmd_opt.txt"
	     INTERRUPTED_STATE_FILE="${CHKP_DIR}/interrupted_state.pth"

	     # Create folders
	     mkdir -p ${CHKP_DIR}
	     mkdir -p ${LOG_DIR}
	     mkdir -p ${VIDEO_DIR}
	     mkdir -p ${RESULTS_DIR}

	     if [ -z "${CHKP_NAME}" ]; then
	         EVAL_CKPT_PATH_DIR="${CHKP_DIR}"
	     else
	         EVAL_CKPT_PATH_DIR="${CHKP_DIR}/ckpt.${CHKP_NAME}.pth"
	     fi

	     # Write commands to file
	     CMD_COMMON_OPTS="--exp-config $EXP_CONFIG_PATH \
	         BASE_TASK_CONFIG_PATH $BASE_TASK_CONFIG_PATH \
	         EVAL_CKPT_PATH_DIR ${EVAL_CKPT_PATH_DIR} \
	         CHECKPOINT_FOLDER ${CHKP_DIR} \
	         TENSORBOARD_DIR ${LOG_DIR} \
	         VIDEO_DIR ${VIDEO_DIR} \
		 	 RESULTS_DIR ${RESULTS_DIR}/{split}/{type} \
		 	 INTERRUPTED_STATE_FILE ${INTERRUPTED_STATE_FILE} \
	         MODEL.RGB_ENCODER.pretrained_ckpt ${REPO_PATH}/data/new_checkpoints/rgb_encoders/${WEIGHTS_NAME} \
		 	 MODEL.DEPTH_ENCODER.ddppo_checkpoint ${REPO_PATH}/data/ddppo-models/gibson-2plus-resnet50.pth \
             MODEL.SEMANTIC_ENCODER.rednet_ckpt ${REPO_PATH}/data/rednet-models/rednet_semmap_mp3d_40_v2_vince.pth \
	         TASK_CONFIG.DATASET.SCENES_DIR ${REPO_PATH}/data/scene_datasets \
	         RL.DDPPO.backbone ${BACKBONE} \
	         TASK_CONFIG.SEED ${SEED} \
	         NUM_UPDATES ${NUM_UPDATES} \
	         VIDEO_OPTION ${VIDEO_OPTION} \
	         ${EXTRA_CMDS}"

	     CMD_TRAIN_OPTS="${CMD_COMMON_OPTS} \
	         TASK_CONFIG.DATASET.SPLIT train \
	         TASK_CONFIG.DATASET.DATA_PATH ${REPO_PATH}/data/datasets/objectnav/objectnav_mp3d/${ENVIRONMENT}/${SPLIT}/${SPLIT}.json.gz \
		 	 NUM_PROCESSES ${NUM_PROCESSES}"
	         #NUM_ENVIRONMENTS ${NUM_ENV} \
	         #WANDB_NAME ${EXP_NAME} \
	         #WANDB_MODE ${WANDB_MODE}"

	     if [ "$RUN_TRAIN_SCRIPT" = true ]; then
	         echo $CMD_TRAIN_OPTS > ${CMD_TRAIN_OPTS_FILE}

	         sbatch \
	             --export=ALL,CMD_OPTS_FILE=${CMD_TRAIN_OPTS_FILE} \
	             --job-name=${EXP_NAME} \
	             --output=$LOG_DIR/log.out \
	             --error=$LOG_DIR/log.err \
	             --partition=$PARTITION \
	             --nodes $NODES \
	             --time $TIME \
	             sbatch_scripts/sbatch_train.sh
	     fi

	     # Evaluate the model simultanously on the saved checkpoints
	     CMD_EVAL_OPTS_FILE="${LOG_DIR}/cmd_eval_opt.txt"

	     CMD_EVAL_OPTS="${CMD_COMMON_OPTS} \
	         EVAL.SPLIT ${VAL_SPLIT} \
	         TASK_CONFIG.DATASET.CONTENT_SCENES [\"*\"] \
	         TEST_EPISODE_COUNT ${TEST_EPISODE_COUNT} \
	         NUM_PROCESSES ${NUM_PROCESSES} \
	         RL.PPO.num_mini_batch 1 \
	         TASK_CONFIG.DATASET.DATA_PATH ${REPO_PATH}/data/datasets/objectnav_mp3d_v2/${VAL_SPLIT}/${VAL_SPLIT}.json.gz \
		 TASK_CONFIG.TASK.TOP_DOWN_MAP.MAP_RESOLUTION 1024"
	         #WANDB_NAME ${EXP_NAME} \
	         #WANDB_MODE ${WANDB_MODE}"
	   
	     # Run evaluation if EVAL_ON_TRAIN is set to True
	     if [ "$RUN_EVAL_SCRIPT" = true ]; then
	         echo "$CMD_EVAL_OPTS" > ${CMD_EVAL_OPTS_FILE}
	       
	         sbatch \
	             --export=ALL,CMD_OPTS_FILE=${CMD_EVAL_OPTS_FILE} \
	             --job-name=${EXP_NAME} \
	             --output=$LOG_DIR/log_${CHKP_NAME}_${VAL_SPLIT}.out \
	             --error=$LOG_DIR/log_${CHKP_NAME}_${VAL_SPLIT}.err \
	             --partition=$PARTITION \
	             --nodes 1 \
	             --time $TIME \
	             sbatch_scripts/sbatch_eval.sh
	     fi

	 }
