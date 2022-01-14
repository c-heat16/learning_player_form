__author__ = 'Connor Heaton'

import os
import json
import torch
import argparse

import numpy as np

from argparse import Namespace
from sklearn.decomposition import PCA


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


def read_data(game_record_dir, batter_form_dir, pitcher_form_dir):
    batter_meta, batter_data = [], []
    pitcher_meta, pitcher_data = [], []
    game_pks = []

    for season in ['2015', '2016', '2017', '2018', '2019']:
        season_dir = os.path.join(game_record_dir, season)
        print('Reading forms for {} season...'.format(season))

        for game_filename in os.listdir(season_dir):
            game_fp = os.path.join(season_dir, game_filename)
            game_j = json.load(open(game_fp))
            game_pk = game_j['game_pk']
            game_pks.append(game_pk)

            for team_side in ['home', 'away']:
                for batter_id in game_j['{}_batting_order'.format(team_side)]:
                    batter_form_fp = os.path.join(batter_form_dir, season, '{}-{}.npy'.format(game_pk, batter_id))
                    batter_form = np.load(batter_form_fp)
                    meta_id = '{}-{}-batter'.format(game_pk, team_side)

                    batter_meta.append(meta_id)
                    batter_data.append(batter_form)

                side_pitcher_id = game_j['{}_starter'.format(team_side)]['__id__']
                pitcher_form_fp = os.path.join(pitcher_form_dir, season, '{}-{}.npy'.format(game_pk, side_pitcher_id))
                pitcher_form = np.load(pitcher_form_fp)
                meta_id = '{}-{}-pitcher'.format(game_pk, team_side)
                pitcher_meta.append(meta_id)
                pitcher_data.append(pitcher_form)

    batter_data = np.vstack(batter_data)
    pitcher_data = np.vstack(pitcher_data)
    print('len(game_pks): {}'.format(len(game_pks)))
    print('len(batter_meta): {} batter_data.shape: {}'.format(len(batter_meta), batter_data.shape))
    print('len(pitcher_meta): {} pitcher_data.shape: {}'.format(len(pitcher_meta), pitcher_data.shape))

    return game_pks, batter_meta, batter_data, pitcher_meta, pitcher_data


def do_pca(data, p=0.95, n=-1):
    pca = PCA(n_components=None)

    all_princomps = pca.fit_transform(data)
    ev_ratios = pca.explained_variance_ratio_
    print('ev_ratios: {}'.format(ev_ratios))
    curr_ev_pct = 0.0
    curr_idx = 0
    if n < 0:
        while curr_ev_pct < p:
            curr_ev_pct += ev_ratios[curr_idx]
            curr_idx += 1
    else:
        curr_idx = n
        print('ev_ratios[:{}]: {}'.format(n, ev_ratios[:n]))

    reqd_comps = all_princomps[:, :curr_idx]
    return reqd_comps


def aggregate_and_make_dict(meta_data, data_arr):
    game_side_data_dict = {}
    game_side_count_dict = {}

    for item_idx, meta_info in enumerate(meta_data):
        item_data = data_arr[item_idx, :].reshape(-1)

        curr_meta_info = game_side_data_dict.get(meta_info, None)
        if curr_meta_info is None:
            curr_meta_info = item_data
        else:
            curr_meta_info += item_data

        game_side_data_dict[meta_info] = curr_meta_info
        game_side_count_dict[meta_info] = game_side_count_dict.get(meta_info, 0) + 1

    for meta_info in game_side_count_dict.keys():
        game_side_data_dict[meta_info] = game_side_data_dict[meta_info] / game_side_count_dict[meta_info]

    return game_side_data_dict


def aggregate_by_game(game_pks, batter_meta, batter_data, pitcher_meta, pitcher_data, out_dir, late_pca, n_pca):
    print('Aggregating batter data...')
    batter_d = aggregate_and_make_dict(batter_meta, batter_data)
    print('Aggregating pitcher data...')
    pitcher_d = aggregate_and_make_dict(pitcher_meta, pitcher_data)
    out_fp_tmplt = os.path.join(out_dir, '{}.npy')

    n_games = 0
    agg_pk_list, agg_reps = [], []
    for game_pk in game_pks:
        game_elements = [
            batter_d['{}-home-batter'.format(game_pk)].reshape(-1),
            pitcher_d['{}-away-pitcher'.format(game_pk)].reshape(-1),
            batter_d['{}-away-batter'.format(game_pk)].reshape(-1),
            pitcher_d['{}-home-pitcher'.format(game_pk)].reshape(-1),
        ]
        game_elements = np.concatenate(game_elements, axis=0)
        agg_pk_list.append(game_pk)
        agg_reps.append(game_elements)

    if late_pca:
        agg_reps = np.vstack(agg_reps)
        print('agg_reps orig shape: {}'.format(agg_reps.shape))
        agg_reps = do_pca(agg_reps, p=0.95, n=n_pca)
        print('agg_reps new shape: {}'.format(agg_reps.shape))

        agg_reps = [agg_reps[i].reshape(-1) for i in range(agg_reps.shape[0])]

    for game_pk, game_elements in zip(agg_pk_list, agg_reps):
        out_fp = out_fp_tmplt.format(game_pk)
        np.save(out_fp, game_elements)
        n_games += 1
        if n_games % 1000 == 0:
            print('Wrote {} game reps to file...'.format(n_games))

    print('Wrote a total of {} game reps to file...'.format(n_games))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records/by_season')
    parser.add_argument('--batter_form_dir',
                        default='/home/czh/sata1/learning_player_form/batter_form_v1')
    parser.add_argument('--pitcher_form_dir',
                        default='/home/czh/sata1/learning_player_form/pitcher_form_v1')
    parser.add_argument('--out',
                        default='/home/czh/sata1/learning_player_form/whole_game_records/by_season_from_form_v1')

    parser.add_argument('--do_pca', default=False, type=str2bool)
    parser.add_argument('--do_late_pca', default=False, type=str2bool)
    parser.add_argument('--n_pca', default=-1, type=int)

    args = parser.parse_args()

    assert not (args.do_pca and args.do_late_pca), 'Can only select one of do_pca and do_late_pca'

    if args.do_pca:
        if args.n_pca > 0:
            outdir = os.path.join(args.out, 'pca-{}'.format(args.n_pca))
        else:
            outdir = os.path.join(args.out, 'pca')
    elif args.do_late_pca:
        if args.n_pca > 0:
            outdir = os.path.join(args.out, 'late_pca-{}'.format(args.n_pca))
        else:
            outdir = os.path.join(args.out, 'late_pca')
    else:
        outdir = os.path.join(args.out, 'raw')

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    args_d = vars(args)
    with open(os.path.join(args.out, 'args.txt'), 'w+') as f:
        for k, v in args_d.items():
            f.write('{} = {}\n'.format(k, v))

    form_data = read_data(args.whole_game_record_dir,
                          args.batter_form_dir,
                          args.pitcher_form_dir)
    all_game_pks, batter_form_info, batter_form_data, pitcher_form_info, pitcher_form_data = form_data

    if args.do_pca:
        # if args.n_pca > 0:
        #     outdir = os.path.join(args.out, 'pca-{}'.format(args.n_pca))
        # else:
        #     outdir = os.path.join(args.out, 'pca')

        print('Performing PCA on batter data...')
        batter_form_data = do_pca(batter_form_data, p=0.95, n=args.n_pca)
        print('\tbatter_form_pca: {}'.format(batter_form_data.shape))

        print('Performing PCA on pitcher data...')
        pitcher_form_data = do_pca(pitcher_form_data, p=0.95, n=args.n_pca)
        print('\tpitcher_form_pca: {}'.format(pitcher_form_data.shape))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print('Aggregating representations by game and saving...')
    aggregate_by_game(all_game_pks, batter_form_info, batter_form_data, pitcher_form_info, pitcher_form_data,
                      outdir, args.do_late_pca, args.n_pca)
