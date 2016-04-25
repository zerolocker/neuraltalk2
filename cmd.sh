#!/bin/sh
#SBATCH -J CS381VpROJ           # Job name
#SBATCH -o experiment.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gpu                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 11:30:00              # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A CS381V-Visual-Recogn         # Specify allocation to charge against

#th eval.lua -model models/model_id1-501-1448236541.t7 -dump_images 0 -image_folder myimg/  -beam_size 3 -num_images -1 -rerank countHighProbObj

th eval.lua -model models/model_id1-501-1448236541.t7 -dump_images 0 -language_eval 1 -input_json coco/cocotalk.json -input_h5 coco/cocotalk.h5  -beam_size 3 -num_images -1 -batch_size 50 -id beam3${1}${2} -reranker_debug_info 0 -rerank countHighProbObj -high_prob_thres ${1} -consider_plural ${2}  >beam3${1}${2}.out

