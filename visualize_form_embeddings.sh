#!/bin/bash

#export FORM_DIR="$PWD/out/forms/batter_form_v1"
#export WHOLE_GAME_DIR="$PWD/data/whole_game_records"

export FORM_DIR="/home/czh/sata1/learning_player_form/forms/batter_form_v1"
export WHOLE_GAME_DIR="/home/czh/sata1/learning_player_form/whole_game_records/by_season"

export DB_FP="$PWD/database/mlb.db"

# move to source dir
cd src

echo "*******************************"
echo "* Visualizing form embeddings *"
echo "*******************************"

python3 visualize_form_embeddings.py --form_rep_dir "$FORM_DIR" --whole_game_records_dir "$WHOLE_GAME_DIR" \
                                     --db_fp "$DB_FP" --n_workers 12 --stats_mode "F"
