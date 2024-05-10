#!/bin/bash
#SBATCH --job-name=lsdp
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --qos=scavenger
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,rlhf.ist.berkeley.edu,cirl.ist.berkeley.edu
#SBATCH --nodes=1

set -x
cd /nas/ucb/$(whoami)/lsdp

eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
conda activate lsdp

LR=${LR:-0.001}
USE_TRANSFORMS=${USE_TRANSFORMS:-False}
LATENT_DIM=${LATENT_DIM:-32}
KL_LOSS_COEFF=${KL_LOSS_COEFF:-0.001}

args=(
    use_transforms=$USE_TRANSFORMS
    lr=$LR
    latent_dim=$LATENT_DIM
    kl_loss_coeff=$KL_LOSS_COEFF
)

python train_vae.py with "${args[@]}" "$@"
