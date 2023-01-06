#!/bin/bash -l
#COBALT -n 1
#COBALT -t 420
#COBALT -q full-node
#COBALT -A superbert

echo loading $1
#module load conda/2021-09-22
module load $1
conda activate

module list

echo python = $(which python)

NODES=`cat $COBALT_NODEFILE | wc -l`
GPUS_PER_NODE=8
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS

export OMP_NUM_THREADS=16

CUDA_VISIBLE_DEVICES=0 python run_nsp.py --pretrained_model_path /grand/projects/SuperBERT/mansisak/semantic_decay_models/checkpoints/1968/Computer-Science/ --start_year 1969 --end_year 2022 --subject Computer-Science >> nsp_cs_results.txt

#Example of how to pick up training from a known checkpoint
#python run_nsp.py --pretrained_model_path /grand/projects/SuperBERT/mansisak/semantic_decay_models/checkpoints/2012/Philosophy/ --start_year 2013 --end_year 2022 --subject Philosophy
