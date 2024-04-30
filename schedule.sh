#!/bin/bash

#SBATCH --job-name=citeseer_tfidf_random25
#SBATCH --output=track_runs/citeseer_tfidf_random25.txt
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=100G
#SBATCH --export=TOKENIZERS_PARALLELISM=false

nvidia-smi
echo $TOKENIZERS_PARALLELISM
# source /home/amangupt/anaconda3/bin/activate 
source /home/amangupt/anaconda3/etc/profile.d/conda.sh
conda activate pgm

./hypertune.sh tfidf