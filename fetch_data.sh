#!/bin/bash

export DB_FP="$PWD/database/mlb.db"

# move to source dir
cd src

echo "******************************"
echo "* Fetching play-by-play data *"
echo "******************************"
python3 fetch_data.py --statcast T --pitching_by_season F --batting_by_season F \
                      --start_year 2015 --end_year 2019 \
                      --n_pybaseball_workers 3 --database_fp $DB_FP

echo "**********************************"
echo "* Fetching season-by-season data *"
echo "**********************************"
python3 fetch_data.py --statcast F --pitching_by_season T --batting_by_season T \
                      --start_year 1995 --end_year 2019 \
                      --n_pybaseball_workers 1 --database_fp $DB_FP
