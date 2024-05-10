#!/bin/bash
#SBATCH --job-name=lsdp
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=ebronstein@berkeley.edu
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --qos=scavenger
##SBATCH --nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,rlhf.ist.berkeley.edu,cirl.ist.berkeley.edu
#SBATCH --nodes=1

set -x
cd /nas/ucb/$(whoami)/lsdp

eval "$(/nas/ucb/ebronstein/anaconda3/bin/conda shell.bash hook)"
conda activate lsdp

SEED=${SEED:-0}

args=(
    seed=$SEED
)

python pusht_state_rollout.py with "${args[@]}" "$@"
