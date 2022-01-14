__author__ = 'Connor Heaton'

import os
import glob
import json
import math
import sqlite3
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def query_db(db_fp, query, args=None):
    conn = sqlite3.connect(db_fp, check_same_thread=False)
    c = conn.cursor()
    if args is None:
        c.execute(query)
    else:
        c.execute(query, args)
    rows = c.fetchall()

    return rows


def read_data(data_dir, reg_game_d):
    game_subpaths, game_seasons = [], []
    n_non_reg_season = 0
    score_diffs = []
    seasons_to_read = ['2015', '2016', '2017', '2018', '2019']

    for season in seasons_to_read:
        print('\tGetting fps for {} season...'.format(season))
        season_dir = os.path.join(data_dir, season)

        for bare_fp in os.listdir(season_dir):
            game_pk = int(bare_fp[:-5])
            if reg_game_d.get(game_pk, False):
                this_subpath = '{}/{}'.format(season, bare_fp)
                game_subpaths.append(this_subpath)
                game_seasons.append(season)

                full_fp = os.path.join(season_dir, bare_fp)
                j = json.load(open(full_fp))
                home_score = j['home_score']
                away_score = j['away_score']
                score_diff = home_score - away_score

                if score_diff < -8:
                    score_diff = -8
                elif score_diff > 8:
                    score_diff = 8

                score_diffs.append(score_diff)
            else:
                n_non_reg_season += 1

    print('* Removed {} non-regular-season games *'.format(n_non_reg_season))

    return game_subpaths, score_diffs


def partition_data(x, p_train=0.8, p_dev=0.1, p_test=0.1, y=None):
    n = len(x)
    idxs = np.array([i for i in range(n)])

    if y is None:
        print('\t*partitioning w/o y*')
        idx_train, idx_test = train_test_split(idxs, train_size=p_train)
        idx_dev, idx_test = train_test_split(idx_test, train_size=p_dev / (p_dev + p_test))
    else:
        print('\t*partitioning w/ y*')
        idx_train, idx_test, y_train, y_test = train_test_split(idxs, y, train_size=p_train, stratify=y)
        idx_dev, idx_test, y_dev, y_test = train_test_split(idx_test, y_test,
                                                            train_size=p_dev / (p_dev + p_test),
                                                            stratify=y_test)

    print('\tidx_train: {}'.format(idx_train.shape))
    print('\tidx_dev: {}'.format(idx_dev.shape))
    print('\tidx_test: {}'.format(idx_test.shape))

    idx_train = list(sorted(idx_train))
    idx_dev = list(sorted(idx_dev))
    idx_test = list(sorted(idx_test))

    train_x = [x[idx] for idx in idx_train]
    dev_x = [x[idx] for idx in idx_dev]
    test_x = [x[idx] for idx in idx_test]

    return train_x, dev_x, test_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--db_fp', default='../database/mlb.db')
    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records/by_season')
    parser.add_argument('--ab_data', default='/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1')
    parser.add_argument('--out', default='../data/whole_game_splits')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print('Finding regular season games...')
    reg_game_query = """select distinct(game_pk) 
                                from statcast
                                where game_year >= 2015 and game_year <= 2019 and game_type = ?"""
    reg_game_args = ('R',)
    reg_game_res = query_db(args.db_fp, reg_game_query, reg_game_args)
    reg_game_d = {rgr[0]: True for rgr in reg_game_res}

    print('Reading data fps...')
    game_fps, score_diffs = read_data(args.whole_game_record_dir, reg_game_d)
    u_scores, u_score_counts = np.unique(score_diffs, return_counts=True)
    comb_score_data = list(zip(u_scores, u_score_counts))
    comb_score_data = sorted(comb_score_data)
    for s, c in comb_score_data:
        print('\tscore diff: {} count: {}'.format(s, c))

    print('Splitting data...')
    train_fps, dev_fps, test_fps = partition_data(game_fps, y=score_diffs)

    print('Writing to file...')
    with open(os.path.join(args.out, 'train.txt'), 'w+') as f:
        f.write('\n'.join(train_fps))

    with open(os.path.join(args.out, 'dev.txt'), 'w+') as f:
        f.write('\n'.join(dev_fps))

    with open(os.path.join(args.out, 'test.txt'), 'w+') as f:
        f.write('\n'.join(test_fps))

