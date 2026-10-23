#!/bin/bash
#SBATCH -J da_purity_lb
#SBATCH -p hx1hdnormal
#SBATCH -A mengjiz
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=dcu:1
#SBATCH -t 06:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.out

set -e
source /public/software/compiler/dtk-25.04.2/env.sh
export ROCM_PATH=/public/software/compiler/dtk-25.04.2
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:/opt/hyhal/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
export DA_EPOCHS=${DA_EPOCHS:-2000}
export DA_LAMBDA_BATCH=${DA_LAMBDA_BATCH:-0.005}

PROJ=$HOME/projects/deepadapter_run
cd $PROJ/downloads/data

echo "node: $(hostname)  date: $(date)  DA_EPOCHS=$DA_EPOCHS  DA_LAMBDA_BATCH=$DA_LAMBDA_BATCH"
$PROJ/conda310/bin/python $PROJ/work/train_tumor.py
echo "exit: $?"
