# learning_player_form
Using advances in machine learning to describe how players impact the game in the MLB

**Before proceeding, please install all packages listed in requirements.txt**


# Implementing Pipeline
1. Fetching data
2. Preparing training data
3. Training player _form_ models
4. Visualizing embeddings

#1. Fetching Data
To fetch data, simply run the `fetch_data.sh` script (i.e. `bash fetch_data.sh`).
By default, this will collect pitch-by-pitch statcast data for 2015-2019, and seasonal statistics back to 1995.
The script will create a `database` directory and create a database file at `database/mlb.db`.

The script utilized two custom "worker" classes, a PyBaseball worker and a SQLWorker.
The PyBaseball workers fetch data via the [PyBaseball](https://github.com/jldbc/pybaseball) library and then pass it to 
an SQLWorker to populate a local database. Pitch-by-pitch data will be collected first, followed by the season-by-season
stats. In total, there will be around 3.6M pitch-by-pitch records, 2k season-by-season pitching records, and 3.7k
season-by-season batting records.
The workers will periodically print their status, ie how many records have been processed and the current date range
being processed. Should only take 10-15 minutes on a modern CPU w/ solid-state storage.



