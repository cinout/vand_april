#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg3|dlg4|dlg5

#SBATCH --job-name="0sh"
#SBATCH --account=punim1623
#SBATCH --time=0-01:30:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=60G

#### export WORLD_SIZE=2   ### FIXME: update world size: nodes x ntasks-per-node
#### export MASTER_PORT=28400
#### echo ">>> NODELIST="${SLURM_NODELIST}
#### master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#### export MASTER_ADDR=$master_addr
#### echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python test.py \
  --mode few_shot \
  --dataset loco \
  --data_path ./data/loco \
  --save_path ./results/loco/few_shot/4shot/seed42 \
  --config_path ./open_clip/model_configs/ViT-L-14-336.json \
  --checkpoint_path ./exps/pretrained/visa_pretrained.pth \
  --model ViT-L-14-336 \
  --features_list 6 12 18 24 \
  --few_shot_features 6 12 18 24 \
  --pretrained openai \
  --image_size 518 \
  --k_shot 4 \
  --seed 42 \

##Log this job's resource usage stats###
my-job-stats -a -n -s
##