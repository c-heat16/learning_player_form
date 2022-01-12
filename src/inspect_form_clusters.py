__author__ = 'Connor Heaton'

import os
import json
import math
import pickle

import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_player_centroids(cluster_membership, player_id=0):
    player_cluster_info = []

    for k, v in cluster_membership.items():
        if k.endswith('.npy'):
            file_info = os.path.basename(k)[:-4]
            this_game_pk, this_player_id = map(int, file_info.split('-'))
        else:
            this_game_pk, this_player_id = map(int, k.split('-'))
        if player_id == this_player_id:
            player_cluster_info.append([this_game_pk, v])

    player_cluster_info = list(sorted(player_cluster_info, key=lambda x: x[0]))
    game_pks, player_clusters = map(list, zip(*player_cluster_info))

    return game_pks, player_clusters


def make_coordinates(all_pks, player_pks, player_centroids):
    player_d = dict(zip(player_pks, player_centroids))

    player_coords = []
    for pk_idx, pk in enumerate(all_pks):
        player_y = player_d.get(pk, None)
        if player_y is not None:
            player_coords.append([pk_idx, player_y])

    xs, ys = map(list, zip(*player_coords))
    return xs, ys


def make_xticks(mapping, all_pks, whole_game_records_dir):
    game_pk_to_season_d = {}
    possible_seasons = ['2015', '2016', '2017', '2018', '2019']
    whole_game_fp_tmplt = os.path.join(whole_game_records_dir, '{}/{}.json')
    for form_filename, form_cluster in mapping.items():
        # print('form_filename: {}'.format(form_filename))
        # season, meta_info = form_filename[:-4].split('/')
        raw_season, meta_info = os.path.split(form_filename[:-4])
        if raw_season.strip() == '':
            game_pk, player_id = map(int, meta_info.split('-'))
            for p_season in possible_seasons:
                if os.path.exists(whole_game_fp_tmplt.format(p_season, game_pk)):
                    season = int(p_season)

        else:
            game_pk, player_id = map(int, meta_info.split('-'))
            season = int(raw_season)
        game_pk_to_season_d[game_pk] = season

    curr_season = None
    tick_idxs, tick_labels = [], []
    for pk_idx, pk in enumerate(all_pks):
        game_season = game_pk_to_season_d[pk]
        if game_season != curr_season:
            tick_idxs.append(pk_idx)
            tick_labels.append(game_season)

        curr_season = game_season

    return tick_idxs, tick_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        default='../out/form_cluster/batter1_agglom/mappings/cluster_map_k75.json')
    parser.add_argument('--out',
                        default='../out/form_cluster/batter1_agglom/eval')
    parser.add_argument('--whole_game_records_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records/by_season')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print('args.data: {}'.format(args.data))
    print('args.out: {}'.format(args.out))
    print('args.whole_game_records_dir: {}'.format(args.whole_game_records_dir))

    n_clusters = int(args.data[:-5].split('k')[-1])
    print('** n_clusters in file: {} **'.format(n_clusters))
    clustering_desc = os.path.split(os.path.split(args.out)[0])[-1]

    data_type = 'Form'
    if 'batter' in args.data:
        player_type = 'batter'
    else:
        player_type = 'pitcher'

    print('** data_type: {} **'.format(data_type))
    print('** player_type: {} **'.format(player_type))
    cluster_mapping = json.load(open(args.data))
    n_form_records = len(cluster_mapping)

    if player_type == 'batter':
        print('Finding centroids for Bryce Harper...')
        harper_pks, harper_clusters = get_player_centroids(cluster_mapping, player_id=547180)
        print('Finding centroids for Mike Trout...')
        trout_pks, trout_clusters = get_player_centroids(cluster_mapping, player_id=545361)
        print('Finding centroids for Giancarlo Stanton...')
        stanton_pks, stanton_clusters = get_player_centroids(cluster_mapping, player_id=519317)
        print('Finding centroids for Neil Walker...')
        walker_pks, walker_clusters = get_player_centroids(cluster_mapping, player_id=435522)

        all_game_pks = list(set(harper_pks) | set(trout_pks) | set(stanton_pks) | set(walker_pks))
        all_game_pks = list(sorted(all_game_pks))
        harper_x, harper_y = make_coordinates(all_game_pks, harper_pks, harper_clusters)
        trout_x, trout_y = make_coordinates(all_game_pks, trout_pks, trout_clusters)
        stanton_x, stanton_y = make_coordinates(all_game_pks, stanton_pks, stanton_clusters)
        walker_x, walker_y = make_coordinates(all_game_pks, walker_pks, walker_clusters)
        xticks, xtick_labels = make_xticks(cluster_mapping, all_game_pks, args.whole_game_records_dir)

        print('# harper_y: {}'.format(len(harper_y)))
        print('\tharper_y[:10]: {}'.format(harper_y[:10]))
        print('# trout_y: {}'.format(len(trout_y)))
        print('\ttrout_y[:10]: {}'.format(trout_y[:10]))
        print('# stanton_y: {}'.format(len(stanton_y)))
        print('\tstanton_y[:10]: {}'.format(stanton_y[:10]))
        print('# walker_y: {}'.format(len(walker_y)))
        print('\twalker_y[:10]: {}'.format(walker_y[:10]))

        min_game_pk = min(min(harper_pks), min(trout_pks), min(stanton_pks), min(walker_pks))

        fig = plt.figure(figsize=(10, 5))
        plt.scatter(stanton_x, stanton_y, c='green', marker=4, alpha=0.5, label='Stanton')
        plt.scatter(walker_x, walker_y, c='orange', marker=5, alpha=0.5, label='Walker')
        plt.scatter(harper_x, [y + 0.0 for y in harper_y], c='red', marker=6, alpha=0.5, label='Harper')
        plt.scatter(trout_x, [y - 0.0 for y in trout_y], c='blue', marker=7, alpha=0.5, label='Trout')
        plt.xticks(xticks, xtick_labels)

        plt.legend(bbox_to_anchor=(1.15, 1))
        plt.title('Batter {}-Cluster Membership Over Time'.format(data_type))
        plt.xlabel('Season')
        plt.ylabel('Cluster ID')
        plt.suptitle(clustering_desc)
        plt.ylim(bottom=-1, top=int(n_clusters))
        plt.savefig(os.path.join(args.out, 'select_batter_{}_cluster_id_over_time_k{}.png'.format(data_type,
                                                                                                  n_clusters)),
                    bbox_inches='tight')
        plt.clf()
    else:
        print('Finding centroids for Gerrit Cole...')
        cole_pks, cole_clusters = get_player_centroids(cluster_mapping, player_id=543037)
        print('Finding centroids for Alex Wood...')
        wood_pks, wood_clusters = get_player_centroids(cluster_mapping, player_id=622072)
        print('Finding centroids for Trevor Bauer...')
        bauer_pks, bauer_clusters = get_player_centroids(cluster_mapping, player_id=545333)
        print('Finding centroids for Justin Verlander...')
        verlander_pks, verlander_clusters = get_player_centroids(cluster_mapping, player_id=434378)

        all_game_pks = list(set(cole_pks) | set(wood_pks) | set(bauer_pks) | set(verlander_pks))
        all_game_pks = list(sorted(all_game_pks))
        cole_x, cole_y = make_coordinates(all_game_pks, cole_pks, cole_clusters)
        wood_x, wood_y = make_coordinates(all_game_pks, wood_pks, wood_clusters)
        bauer_x, bauer_y = make_coordinates(all_game_pks, bauer_pks, bauer_clusters)
        verlander_x, verlander_y = make_coordinates(all_game_pks, verlander_pks, verlander_clusters)
        xticks, xtick_labels = make_xticks(cluster_mapping, all_game_pks)

        print('# cole clusters: {}'.format(len(cole_clusters)))
        print('\tcole_clusters[:10]: {}'.format(cole_clusters[:10]))
        print('# wood clusters: {}'.format(len(wood_clusters)))
        print('\twood_clusters[:10]: {}'.format(wood_clusters[:10]))
        print('# bauer clusters: {}'.format(len(bauer_clusters)))
        print('\tbauer_clusters[:10]: {}'.format(bauer_clusters[:10]))
        print('# verlander clusters: {}'.format(len(verlander_clusters)))
        print('\tverlander_clusters[:10]: {}'.format(verlander_clusters[:10]))

        min_game_pk = min(min(cole_pks), min(wood_pks), min(bauer_pks), min(verlander_pks))

        fig = plt.figure(figsize=(10, 5))
        plt.scatter(wood_x, wood_y, c='green', marker=4, alpha=0.5, label='Wood')
        plt.scatter(verlander_x, verlander_y, c='orange', marker=5, alpha=0.5, label='Verlander')
        plt.scatter(bauer_x, [y + 0.12 for y in bauer_y], c='red', marker=6, alpha=0.5, label='Bauer')
        plt.scatter(cole_x, [y - 0.12 for y in cole_y], c='blue', marker=7, alpha=0.5, label='Cole')

        plt.xticks(xticks, xtick_labels)
        plt.legend(bbox_to_anchor=(1.17, 1))
        plt.title('Pitcher {}-Cluster Membership Over Time'.format(data_type))
        plt.xlabel('Season')
        plt.ylabel('Cluster ID')
        plt.suptitle(clustering_desc)
        plt.ylim(bottom=-1, top=int(n_clusters))
        plt.savefig(os.path.join(args.out, 'select_pitcher_{}_cluster_id_over_time_k{}.png'.format(data_type,
                                                                                                   n_clusters)),
                    bbox_inches='tight')
        plt.clf()








