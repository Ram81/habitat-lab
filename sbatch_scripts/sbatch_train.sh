#!/bin/bash
#SBATCH --job-name=ddppo-nav
#SBATCH --output=/checkpoint/%u/jobs/job.%A_%a.out
#SBATCH --error=/checkpoint/%u/jobs/job.%A_%a.err
#SBATCH --gpus-per-task 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=devlab
#SBATCH --time=72:00:00
#SBATCH --signal=SIGUSR1@300
#SBATCH --open-mode=append
#SBATCH --constraint=volta32gb
####SBATCH --array=0-4

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda deactivate
conda activate habitat_objnav # /private/home/alexclegg/.conda/envs/habitat_il

module purge
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load NCCL/2.7.8-1-cuda.11.0

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}
export GLOG_minloglevel=3
export MAGNUM_LOG=quiet

#To avoid hanging with distrib.init
# export NCCL_SOCKET_IFNAME=“”
# export GLOO_SOCKET_IFNAME=“”

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export PYTHONPATH="/private/home/alexclegg/karmesh_code/objectnav/habitat-sim/"

set -x
repeat_number=${SLURM_ARRAY_TASK_ID}
CMD_OPTS=$(cat "$CMD_OPTS_FILE")
CURRENT_DATETIME="`date +%Y-%m-%d_%H-%M-%S`";

echo "Cuda Visible Devices: " $CUDA_VISIBLE_DEVICES
echo "Date: " $CURRENT_DATETIME
echo "Commands Provided: " $CMD_OPTS
echo "MAIN Address: " $MAIN_ADDR

printenv | grep SLURM
#cd habitat_baselines/
srun -u --kill-on-bad-exit=1 \
    python -u -m habitat_baselines.run \
    --run-type train \
    ${CMD_OPTS}

