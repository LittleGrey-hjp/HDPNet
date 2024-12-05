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
srun --nodelist=c01 --export=ALL,CUDA_VISIBLE_DEVICES=3 --pty python /share/home/project/HDPNet/train.py --path "/share/home/dataset/COD-TrainDataset/" --pretrain "/share/home/project/HDPNet/pvt_v2_b5_22k_30ep.pth"
echo "end $(date)"
