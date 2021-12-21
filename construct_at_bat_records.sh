#!/bin/bash

#export OUT_DIR="$PWD/data/ab_seqs/ab_seqs_v1"
export OUT_DIR="/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1"
export DB_FP="../database/mlb.db"
export N_WORKERS=16

# move to source dir
cd src

python3 construct_at_bat_records.py --start_year 2015 --end_year 2019 --n_workers $N_WORKERS \
                                    --out $OUT_DIR --db_fp $DB_FP