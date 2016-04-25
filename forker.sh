#!/bin/bash

for high_prob_thres in 0.001 0.01 0.05 0.1 0.2; do
for consider_plural in 'Plu' 'noPlu'; do
sbatch cmd.sh ${high_prob_thres} ${consider_plural}
done
done
