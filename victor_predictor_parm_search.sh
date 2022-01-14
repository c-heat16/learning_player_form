#!/bin/bash

export DB_FP="$PWD/database/mlb.db"

# Inputs
export AB_SEQ_DIR="$PWD/data/ab_seqs/ab_seqs_v1"
export WHOLE_GAME_RECORD_DIR="$PWD/data/whole_game_records/by_season"
export BATTER_FORM_DIR="$PWD/out/forms/batter_form_v1"
export PITCHER_FORM_DIR="$PWD/out/forms/pitcher_form_v1"

#export AB_SEQ_DIR="/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1"
#export WHOLE_GAME_RECORD_DIR="/home/czh/sata1/learning_player_form/whole_game_records/by_season"
#export BATTER_FORM_DIR="/home/czh/sata1/SportsAnalytics/batter_form_v1"
#export PITCHER_FORM_DIR="/home/czh/sata1/SportsAnalytics/pitcher_form_v1"

# Outputs
export SPLITS_DIR="$PWD/data/whole_game_splits"
export FORM_VECTOR_DIR="$PWD/data/game_form_vectors_v1"
export TEAM_META_VECTOR_DIR="$PWD/data/game_meta_vectors_v1"
export PARM_SEARCH_OUT_DIR="$PWD/out/basic_parm_search"

# move to source dir
cd src

echo "************************"
echo "* Creating Game Splits *"
echo "************************"
python3 create_game_splits.py --db_fp "$DB_FP" --whole_game_record_dir "$WHOLE_GAME_RECORD_DIR" \
                              --ab_data "$AB_SEQ_DIR" --out "$SPLITS_DIR"

echo "******************************"
echo "* Creating Team Meta Vectors *"
echo "******************************"
python3 construct_game_meta_vectors.py --db_fp "$DB_FP" --out "$TEAM_META_VECTOR_DIR" \
                                       --whole_game_record_dir "$WHOLE_GAME_RECORD_DIR"

echo "******************************"
echo "* Creating Team Stat Vectors *"
echo "******************************"
python3 construct_game_stat_vectors.py --whole_game_record_dir "$WHOLE_GAME_RECORD_DIR" \
                                       --splits_basedir "$SPLITS_DIR"

echo "******************************"
echo "* Creating Game Form Vectors *"
echo "******************************"
python3 create_game_reps_from_form.py --whole_game_record_dir "$WHOLE_GAME_RECORD_DIR" \
                                      --batter_form_dir "$BATTER_FORM_DIR" \
                                      --pitcher_form_dir "$PITCHER_FORM_DIR" \
                                      --out "$FORM_VECTOR_DIR" \
                                      --do_pca True --n_pca 5

echo "********************************"
echo "* Performing Basic Parm Search *"
echo "********************************"
python3 basic_parm_search.py --use_stats T --use_form T --use_meta T \
                             --do_rf T --do_logreg F --do_svm F  \
                             --whole_game_record_dir "$WHOLE_GAME_RECORD_DIR" \
                             --splits_basedir "$SPLITS_DIR" \
                             --form_dir "$FORM_VECTOR_DIR" --form_subdir "pca-5" \
                             --game_meta_fp "$TEAM_META_VECTOR_DIR/game_meta_vectors.json" \
                             --out "$PARM_SEARCH_OUT_DIR" --force_new_data T

