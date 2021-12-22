#!/bin/bash

#export FORM_OUT_DIR="$PWD/data/"
#export AB_OUT_DIR="$PWD/data/ab_seqs/ab_seqs_v1"
#export CAREER_OUT_DIR="$PWD/data/player_career_data"
#export WHOLE_GAME_OUT_DIR="$PWD/data/whole_game_records"

export FORM_OUT_DIR="/home/czh/sata1/learning_player_form/forms"
export AB_OUT_DIR="/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1"
export CAREER_OUT_DIR="/home/czh/sata1/learning_player_form/player_career_data"
export WHOLE_GAME_OUT_DIR="/home/czh/sata1/learning_player_form/whole_game_records"

export BATTER_MODEL_FP="$PWD/pretrained_models/batter_form_model/models/model_370e.pt"
export PITCHER_MODEL_FP="$PWD/pretrained_models/pitcher_form_model/models/model_175e.pt"

# move to source dir
cd src

echo "***************************"
echo "* Describing player forms *"
echo "***************************"

python3 describe_player_forms.py --ab_data "$AB_OUT_DIR" --career_data "$CAREER_OUT_DIR" \
                                 --whole_game_record_dir "$WHOLE_GAME_OUT_DIR" \
                                 --model_ckpt "$BATTER_MODEL_FP" --out "$FORM_OUT_DIR" \
                                 --start_year 2015 --end_year 2019 \
                                 --n_workers -1 --out_dir_tmplt "{}_form_v1"