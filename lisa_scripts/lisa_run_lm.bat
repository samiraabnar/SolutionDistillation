#!/bin/bash

#SBATCH --job-name=lm_ptb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

module load surf-devel
module load 2019
module load Python/3.6.6-intel-2018b
module load CUDA/10.0.130
module load cuDNN/7.3.1-CUDA-10.0.130
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian/cuDNN/7.3.1-CUDA-10.0.130/lib64:$LD_LIBRARY_PATH

#cd /home/samigpu/Codes/SolutionDistillation
source /home/samigpu/my_env/bin/activate

HEAD_DIR="/home/samigpu/Codes/SolutionDistillation"
CODE_DIR=$HEAD_DIR/distill
DATA_DIR=$HEAD_DIR/data
LOGS_DIR=$HEAD_DIR/logs

#mkdir "$TMPDIR"/samigpu

TMP_DATA_DIR=$DATA_DIR
#"$TMPDIR"/samigpu/data
TMP_LOGS_DIR=$LOGS_DIR
#"$TMPDIR"/samigpu/logs

#mkdir $TMP_DATA_DIR
#mkdir $TMP_LOGS_DIR

#Copy input file to scratch
#cp -r "$DATA_DIR"/* "$TMP_DATA_DIR"/
#cp -r "$LOGS_DIR"/* "$TMP_LOGS_DIR"/

echo "$TMP_LOGS_DIR"
echo "$TMP_DATA_DIR"
export PYTHONPATH=$PYTHONPATH:$HEAD_DIR

CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/lm_trainer.py \
--log_dir="$TMP_LOGS_DIR" --data_dir="$TMP_DATA_DIR" \
--batch_size=32 --hidden_dim=728 --embedding_dim=728 \
--learning_rate=0.001 \
--hidden_dropout_keep_prob=0.8 --input_dropout_keep_prob=0.6 \
--task_name=sent_wiki \
--exp_name b32_drop8-6

#Copy input file to scratch
#cp -r  "$TMP_DATA_DIR"/* "$DATA_DIR"/
#cp -r  "$TMP_LOGS_DIR"/* "$LOGS_DIR"/

