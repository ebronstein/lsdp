#!/bin/bash
#SBATCH --job-name=lsdp
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --qos=scavenger
#SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,rlhf.ist.berkeley.edu,cirl.ist.berkeley.edu
#SBATCH --nodes=1

set -x
cd /nas/ucb/$(whoami)/lsdp

eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
conda activate lsdp

OBS_KEY=${OBS_KEY:-"img"}
N_EPOCHS=${N_EPOCHS:-250}
VAE_MODEL_PATH=${VAE_MODEL_PATH:-"/nas/ucb/ebronstein/lsdp/models/pusht_vae/vae_32_20240403.pt"}

# TODO
args=(
    obs_key=$OBS_KEY
    n_epochs=$N_EPOCHS
    vae_model_path=$VAE_MODEL_PATH
)

python state_diffusion.py with "${args[@]}" "$@"
