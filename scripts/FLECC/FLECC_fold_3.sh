#!/bin/bash
# Job name:
#SBATCH --job-name=ap421TrainJob
#
# Partition:
#SBATCH --partition=gpu-serv-02-part
#
# Specify one task:
#SBATCH --ntasks-per-node=1
#
# Number of processors for single task needed for use case (example):
#SBATCH --cpus-per-task=4
#
#SBATCH --mem=16GB
#
#SBATCH --gpus-per-task=1
#
#SBATCH --gpu-bind=single:1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
## Command(s)

echo "fold 3"
dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_FLECC-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/FLECC/fold_3_mult_2.json --save /data/ap421/weights/FLECC/FLECCs_fold_3_competitive --fold 3 --multiplier 2

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_FLECC-2024-05-16.json --batch_size 32 --epochs 3 --learning_rate 1e-4 --history /data/ap421/history/FLECC/fold_3_mult_1_init.json --save /data/ap421/weights/FLECC/FLECCs_fold_3_competitive --fold 3 --multiplier 1 --pre_trained /data/ap421/weights/FLECC/FLECCs_fold_3_competitive_final

dnn-env/bin/python /data/ap421/EFE_Project/network/competitive_network.py --dataset /data/ap421/EFE_Project/data/datasets/dataset_FLECC-2024-05-16.json --batch_size 32 --epochs 4 --learning_rate 1e-5 --history /data/ap421/history/FLECC/fold_3_mult_1_fin.json --save /data/ap421/weights/FLECC/FLECCs_fold_3_competitive --fold 3 --multiplier 1 --pre_trained /data/ap421/weights/FLECC/FLECCs_fold_3_competitive_final