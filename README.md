# Learning to Describe How Players Impact the Game in the MLB
This repository contains the elements necessary to describe how players in the MLB impact the game over the short term,
what we colloquially refer to as their _form_. The steps described below will guide you through the process from start
to finish - collecting the data, training the model, and visualizing player embeddings. Examples of the embedding
visualizations are given below.

# Example Batter Embeddings
Below is an example of embeddings that were obtained from a batter _form_ model trained following the process described 
below. A more complete set of embedding visualizations can be found [here](figures/batter_form_plots.png).
![Example batter form embeddings](figures/succinct_batter_form_plots.png)

# Example Pitcher Embeddings
Below is an example of embeddings that were obtained from a pitcher _form_ model trained following the process described 
below. A more complete set of embedding visualizations can be found [here](figures/pitcher_form_plots.png).
![Example pitcher form embeddings](figures/succinct_pitcher_form_plots.png)


# Implementing Pipeline

**Before proceeding, please install all packages listed in requirements.txt**

1. Fetching data
2. Preparing training data
3. Training player _form_ models
4. Describing player _form_
5. Visualizing embeddings

# 1. Fetching Data

To fetch data, simply run the [`fetch_data.sh`](fetch_data.sh) script (i.e. `bash fetch_data.sh`).
By default, this will collect pitch-by-pitch statcast data for 2015-2019, and seasonal statistics back to 1995.
The script will create a `database` directory and create a database file at `database/mlb.db`. To change where the
database is placed, change the `DB_FP` variable in the `fetch_data.sh` script to the desired location. If you change the
location of the database, please make note of it as you will need it later.

The script utilized two custom "worker" classes, a PyBaseball worker and a SQLWorker.
The PyBaseball workers fetch data via the [PyBaseball](https://github.com/jldbc/pybaseball) library and then pass it to 
an SQLWorker to populate a local database. Pitch-by-pitch data will be collected first, followed by the season-by-season
stats. In total, there will be around 3.6M pitch-by-pitch records, 2k season-by-season pitching records, and 3.7k
season-by-season batting records.
The workers will periodically print their status, ie how many records have been processed and the current date range
being processed. Should only take 10-15 minutes on a modern CPU w/ solid-state storage.

# 2. Preparing Training Data

The first step in creating the training data is to make a single record for each plate appearance in the newly 
constructed database. To do so, simply run the [`construct_at_bat_records.sh`](construct_at_bat_records.sh) script. 
If you created the database in a location other than `database/mlb.db` (i.e. you changed `DB_FP` in the `fetch_data.sh` 
script), please update that for this script as well. By default, running the script will create a `data/` directory in 
the repo, and the individual at-bat records will be placed in the `data/ab_seqs/ab_seqs_v1/` directory, grouped by 
season. The output location can be adjusted by modifying the `AB_OUT_DIR` variable in the script.

In a system with a modern CPU and solid-state storage, it will take roughly 4.5 seconds to construct the at-bats for an
individual game. By default, the script will try to utilize 4 threads to construct the records. This can be adjusted
by changing the value of the `N_WORKERS` variable in the `construct_at_bat_records.sh` script. When 16 threads are used,
it will take roughly 1 hours for the script to complete processing. Records for each season will take up about 10.5 GB,
so for all five seasons, so ~55 GB of free space is required.

Once the at-bat records are constructed, the script will begin to aggregate at-bat records by player in chronological
order. First pitchers, then batters. This should only take about a minute or less. By default, the script will create a
`data/player_career_data` directory in the repo and place the output within. If you wish to change the location of the
output, simply change the `CAREER_OUT_DIR` variable in the script.

Finally, the script will create whole game records that will be used later on. That is, records describing individual 
games in terms of the starting batters, pitchers, location, score, and hits among others. By default, the records will
be placed in the `data/whole_game_records/by_season` directory in the repo. This phase should only take around five 
minutes.

# 3. Training Player _Form_ Models
**NOTE:** Before proceeding with this step, please unsure you have a CUDA capable GPU and CUDA installed on the machine
you intend to run the models on. Additional information to this end can be found 
[here](https://pytorch.org/get-started/locally/). While you can _technically_ train these models on a CPU, it would take
an egregious amount of time to complete.

**NOTE:** GPUs with a minimum **32 GB** of RAM are required to *train* the batter and pitcher models referenced in our
corresponding paper (defaults in training scripts). One GPU was used to train the batter model, while two GPUs were 
used to train the pitcher model.

We provide scripts to train both the batter and pitcher _form_ models as presented in our paper. As their names suggest,
[`batter_form_modeling.sh`](batter_form_modeling.sh) trains a batter _form_ model and 
[`pitcher_form_modeling.sh`](pitcher_form_modeling.sh) trains a pitcher _form_ model. The batter model will take a 
litte more than 3 days to train (~80 hours) while the pitcher model will take about 2.5 days to train (~58 hours) using 
A6000 GPU's. We also provide the trained model weights in the [`pretrained_models`](pretrained_models) directory in the
repo.

# 4. Describing Player _Form_
The [`describe_player_forms.sh`](describe_player_forms.sh) script is provided for you to describe the form of players in
the starting lineup for games from 2015-2019. Please remember to update the `FORM_OUT_DIR`, `AB_OUT_DIR`, 
`CAREER_OUT_DIR`, and `WHOLE_GAME_OUT_DIR` variables in the script if you have changed them in any of the previous 
scripts. If ran as given, the script will describe **batter** form using the provided pretrained batter model. To 
describe **pitcher** form, change the `--model_ckpt "$BATTER_MODEL_FP"` script argument to 
`--model_ckpt "PITCHER_MODEL_FP"`.  If you wish to use a different model, simply provide the filepath to the desired
checkpoint instead.

**NOTE:** If providing a different model checkpoint, the script expects the parameters defining the model to be in an 
`args.txt` in the parent directory of the model checkpoint file. For example, if the path to the model checkpoint is
`model_time_id/models/model_ckpt.pt`, the script will expect the model parameters to be found in 
`model_time_id/args.txt`.

When using the script as provided (`--n_workers -1`), the script will use 10 threads to build the input data for 
batters and 3 threads for pitchers. With these parameters, it will take ~15 minutes to process one season of batters
and ~7 minutes for a season on pitchers. If you wish to use a different number of threads, change `--n_workers` to the 
desired value.




















