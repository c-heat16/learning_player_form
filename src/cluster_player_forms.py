__author__ = 'Connor Heaton'

import os
import json
import argparse
import numpy as np

from sklearn.cluster import AgglomerativeClustering


def read_data(data_dir):
    all_fps, all_embds = [], []
    for season in os.listdir(data_dir):
        if season.startswith('2'):
            print('Reading form embds for {} season...'.format(season))
            season_dir = os.path.join(data_dir, season)
            for form_filename in os.listdir(season_dir):
                form_fp = os.path.join(season_dir, form_filename)
                effective_fp = '{}/{}'.format(season, form_filename)
                if form_fp.endswith('npy'):
                    form_embd = np.load(form_fp, allow_pickle=True)
                    all_fps.append(effective_fp)
                    all_embds.append(form_embd)

    all_embds = np.vstack(all_embds)
    return all_fps, all_embds


def agglom_cluster_data(embds, k):
    print('Fitting AgglomerativeClustering model w/ k={}...'.format(k))

    model = AgglomerativeClustering(n_clusters=k).fit(embds)

    embd_clstr_labels = model.labels_

    return embd_clstr_labels, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/czh/sata1/learning_player_form/forms/batter_form_v1')
    parser.add_argument('--out', default='../out/form_cluster/batter1_agglom')
    args = parser.parse_args()

    form_cluster_basedir = os.path.split(args.out)[0]
    if not os.path.exists(form_cluster_basedir):
        os.makedirs(form_cluster_basedir)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    cluster_map_out_fp_tmplt = os.path.join(args.out, 'mappings', 'cluster_map_k{}.json')
    if not os.path.exists(os.path.join(args.out, 'mappings')):
        os.makedirs(os.path.join(args.out, 'mappings'))

    if 'batter' in args.data:
        ks = [75, 50, 25]
    else:
        ks = [32, 16, 8]

    form_fps, form_embds = read_data(args.data)
    print('len(form_fps): {}'.format(len(form_fps)))
    print('form_embds: {}'.format(form_embds.shape))
    for k in ks:
        embd_clstr_ids, model = agglom_cluster_data(form_embds, k)

        print('Creating and saving cluster mapping...')
        clstr_map = {}
        for i in range(embd_clstr_ids.shape[0]):
            clstr_map[form_fps[i]] = int(embd_clstr_ids[i])

        map_out_fp = cluster_map_out_fp_tmplt.format(k)
        print('\tmap_out_fp: {}'.format(map_out_fp))
        with open(map_out_fp, 'w+') as f:
            f.write(json.dumps(clstr_map, indent=2))

