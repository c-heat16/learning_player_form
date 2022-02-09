#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

export AB_OUT_DIR="$PWD/data/ab_seqs/ab_seqs_v1"
export CAREER_OUT_DIR="$PWD/data/player_career_data"
#export AB_OUT_DIR="/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1"
#export CAREER_OUT_DIR="/home/czh/sata1/learning_player_form/player_career_data"

cd src/

python3 run_player_form_modeling.py --player_type "pitcher" --epochs 175 --save_model_every 5 \
                                    --batch_size 48 \
                                    --min_view_step_size 1 --max_view_step_size 15 --view_size 60 \
                                    --form_ab_window_size 75 --min_form_ab_window_size 70 \
                                    --min_ab_to_be_included_in_dataset 100 \
                                    --max_seq_len 550 --max_view_len 420 \
                                    --distribution_based_player_sampling_prob 0.25 \
                                    --mask_override_prob 0.15 --n_warmup_iters 4000 \
                                    --n_data_workers 4 --gpus 0 1 --port 12345 \
                                    --n_layers 8 --n_attn 8 --n_proj_layers 2 \
                                    --ab_data "$AB_OUT_DIR" --career_data "$CAREER_OUT_DIR"