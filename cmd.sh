#!/bin/sh
#SBATCH -J CS381VpROJ           # Job name
#SBATCH -o experiment.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gpu                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 11:30:00              # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A CS381V-Visual-Recogn         # Specify allocation to charge against

#th eval.lua -model model_id1-501-1448236541.t7 -dump_images 0 -image_folder myimg/  -beam_size 3 -num_images -1 -rerank rank1+rank2

#th ori_eval.lua -model model_id1-501-1448236541.t7 -dump_images 0 -language_eval 1 -input_json coco/cocotalk.json -input_h5 coco/cocotalk.h5   -beam_size 10 -num_images -1 -batch_size 100 -id beam10 > ori_beam10.out

#th eval.lua -model model_id1-501-1448236541.t7 -dump_images 0 -language_eval 1 -input_json coco/cocotalk.json -input_h5 coco/cocotalk.h5  -beam_size 3 -num_images -1 -batch_size 80 -id beam3rerankp1p2 -reranker_debug_info 0 -rerank P1+P2 >beam3p1p2.out
#th eval.lua -model model_id1-501-1448236541.t7 -dump_images 0 -language_eval 1 -input_json coco/cocotalk.json -input_h5 coco/cocotalk.h5  -beam_size 3 -num_images -1 -batch_size 80 -id beam3rerankr1r2 -reranker_debug_info 0 -rerank rank1+rank2 >beam3r1r2.out


th eval.lua -model model_id1-501-1448236541.t7 -dump_images 0 -language_eval 1 -input_json coco/cocotalk.json -input_h5 coco/cocotalk.h5  -beam_size 10 -num_images -1 -batch_size 80 -id beam10rerankp1p2 -reranker_debug_info 0 -rerank P1+P2 >beam10p1p2.out
#th eval.lua -model model_id1-501-1448236541.t7 -dump_images 0 -language_eval 1 -input_json coco/cocotalk.json -input_h5 coco/cocotalk.h5  -beam_size 10 -num_images -1 -batch_size 80 -id beam3rerankr1r2 -reranker_debug_info 0 -rerank rank1+rank2 >beam10r1r2.out

