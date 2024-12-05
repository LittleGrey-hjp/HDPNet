#!/bin/bash
#SBATCH -J COD #作业名字（⾮必需）
#SBATCH -o %j.out.txt #标准输出⽂件（⾮必需）
#SBATCH -e %j.err.txt #标准错误输出⽂件（⾮必需）
#SBATCH -p gpu #运⾏分区（⾮必需，如不指定分配到默认分区）
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

source /share/home/anaconda3/etc/profile.d/conda.sh
conda activate torch_mmcd

echo "start $(date)"
### the command to run
srun --nodelist=c01 --pty python /share/home/project/HDPNet/test.py
echo "end $(date)"
