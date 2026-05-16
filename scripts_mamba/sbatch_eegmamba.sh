#!/bin/bash
#SBATCH --job-name=eegmamba_icare
#SBATCH --output=eegmamba_%j.out
#SBATCH --error=eegmamba_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00              
#SBATCH --partition=gpu            

# 0. Navigate to the project root
cd /scratch/lustre/home/maza9905/eegmamba_folder/

echo "Setting up environment..."

# 1. Initialize Conda Environment
export PATH=~/miniconda3/bin:$PATH
. activate base
conda activate eegmamba

# 2. Export Required Library Paths
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# 3. Environment Optimizations
export TF_CPP_MIN_LOG_LEVEL=2        # Reduce noise
export PYTHONUNBUFFERED=1

echo "Starting EEGMamba fine-tuning on I-CARE dataset..."

# 4. Run the Script
# Note: Ensure datasets_dir points to your LMDB database on the HPC
python EEGMamba/finetune_main.py \
    --downstream_dataset ICARE \
    --datasets_dir /scratch/lustre/home/maza9905/eegmamba_folder/icare_data/processed_mamba/ \
    --model_dir /scratch/lustre/home/maza9905/eegmamba_folder/EEGMamba/models \
    --foundation_dir /scratch/lustre/home/maza9905/eegmamba_folder/EEGMamba/pretrained_weights/pretrained_EEGMamba.pth \
    --batch_size 32 \
    --epochs 20 \
    --patience 5 \
    --num_workers 10

echo "Job complete!"
