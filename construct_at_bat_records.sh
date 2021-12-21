#!/bin/bash

#export AB_OUT_DIR="$PWD/data/ab_seqs/ab_seqs_v1"
#export CAREER_OUT_DIR="$PWD/data/player_career_data"
export AB_OUT_DIR="/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1"
export CAREER_OUT_DIR="/home/czh/sata1/learning_player_form/player_career_data"

export DB_FP="$PWD/database/mlb.db"
export N_WORKERS=16

# move to source dir
cd src

echo "Building at-bat records..."
python3 construct_at_bat_records.py --start_year 2015 --end_year 2019 --n_workers $N_WORKERS \
                                    --out AB_OUT_DIR --db_fp "$DB_FP"

echo "Building pitcher career data..."
python3 construct_player_career_records.py --player_type "pitcher" --db_fp "$DB_FP" --outdir "$CAREER_OUT_DIR"

echo "Building batter career data..."
python3 construct_player_career_records.py --player_type "batter" --db_fp "$DB_FP" --outdir "$CAREER_OUT_DIR"