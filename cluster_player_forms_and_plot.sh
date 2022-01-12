#!/bin/bash

export FORM_DIR="$PWD/out/forms/batter_form_v1"
#export FORM_DIR="$PWD/out/forms/pitcher_form_v1"

export CLUSTER_OUT_DIR="$PWD/out/form_cluster/batter1_agglom"
#export CLUSTER_OUT_DIR="$PWD/out/form_cluster/pitcher1_agglom"

export WHOLE_GAME_RECORD_DIR="$PWD/data/whole_game_records/by_season"
#export WHOLE_GAME_RECORD_DIR="/home/czh/sata1/learning_player_form/whole_game_records/by_season"

export CLUSTER_TO_PLOT="$CLUSTER_OUT_DIR/mappings/cluster_map_k75.json"
export FIG_OUT_DIR="$CLUSTER_OUT_DIR/eval"

# move to source dir
cd src

echo "***************************"
echo "* Clustering player forms *"
echo "***************************"
python3 cluster_player_forms.py --data "$FORM_DIR" --out "$CLUSTER_OUT_DIR"

echo "**************************"
echo "* Plotting form clusters *"
echo "**************************"
python3 inspect_form_clusters.py --data "$CLUSTER_TO_PLOT" --out "$FIG_OUT_DIR" \
                                 --whole_game_records_dir "$WHOLE_GAME_RECORD_DIR"