#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export AB_OUT_DIR="$PWD/data/ab_seqs/ab_seqs_v1"
export CAREER_OUT_DIR="$PWD/data/player_career_data"
#export AB_OUT_DIR="/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1"
#export CAREER_OUT_DIR="/home/czh/sata1/learning_player_form/player_career_data"

cd src/

python3 run_player_form_modeling.py --player_type "batter" --epochs 370 --batch_size 256 \
                                    --min_view_step_size 1 --max_view_step_size 5 --view_size 15 \
                                    --form_ab_window_size 20 --min_form_ab_window_size 20 \
                                    --min_ab_to_be_included_in_dataset 40 \
                                    --max_seq_len 200 --max_view_len 125 \
                                    --distribution_based_player_sampling_prob 0.25 \
                                    --mask_override_prob 0.15 --n_warmup_iters 2000 \
                                    --n_data_workers 4 --gpus 0 --port 12345 \
                                    --ab_data "$AB_OUT_DIR" --career_data "$CAREER_OUT_DIR"