__author__ = 'Connor Heaton'

import os
import json
import math
import time
import sqlite3
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from multiprocessing import Process, Manager


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


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("huh?")


def autoconvert(s):
    if s in ['[BOS]', '[EOS]']:
        return s
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass

    if s[0] == '[' and s[-1] == ']':
        s = s[1:-1]
        s = [ss.strip('\'') for ss in s.split(',')]

    return s


def read_model_args(fp):
    m_args = {}

    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '':
                arg, val = line.split('=')
                arg = arg.strip()
                val = val.strip()

                val = autoconvert(val)
                m_args[arg] = val

    # m_args = Namespace(**m_args)

    return m_args


class EmbdReader(object):
    def __init__(self, meta_list, out_q, worker_idx):
        self.meta_list = meta_list
        self.out_q = out_q
        self.worker_idx = worker_idx

    def work(self):
        for meta_idx, meta_item in enumerate(self.meta_list):
            form_fp, whole_game_fp, player_id = meta_item
            form = np.load(form_fp)
            whole_game_j = json.load(open(whole_game_fp))

            if player_id in whole_game_j['home_batting_order']:
                batting_order = whole_game_j['home_batting_order'].index(player_id)
            elif player_id in whole_game_j['away_batting_order']:
                batting_order = whole_game_j['away_batting_order'].index(player_id)
            else:
                batting_order = -1

            self.out_q.put([player_id, form, batting_order])

            if meta_idx % 5000 == 0:
                print('** Reader {} read {} items **'.format(self.worker_idx, meta_idx))

        print('EmbdReader {} pushing term item...'.format(self.worker_idx))
        self.out_q.put('[TERM]')


def read_embds(form_rep_dir, whole_game_records_dir, n_workers=4):
    fp_meta = []
    print('Reading filepaths...')
    for season in os.listdir(form_rep_dir):
        if season.startswith('2'):
            season_dir = os.path.join(form_rep_dir, season)

            for filename in os.listdir(season_dir):
                if filename.endswith('.npy'):
                    form_fp = os.path.join(season_dir, filename)
                    game_pk, player_id = map(int, filename[:-4].split('-'))
                    whole_game_fp = os.path.join(whole_game_records_dir, season, '{}.json'.format(game_pk))

                    this_meta = [form_fp, whole_game_fp, player_id]
                    fp_meta.append(this_meta)

    print('Configuring workers...')
    chunk_size = int(len(fp_meta) / n_workers)
    meta_by_worker = []
    for i in range(n_workers):
        if i < n_workers - 1:
            this_worker_meta = fp_meta[:chunk_size]
            fp_meta = fp_meta[chunk_size:]

            meta_by_worker.append(this_worker_meta)
        else:
            meta_by_worker.append(fp_meta[:])

    for idx, meta in enumerate(meta_by_worker):
        print('idx: {} len(meta): {}'.format(idx, len(meta)))

    m = Manager()
    read_data_q = m.Queue()
    readers = [EmbdReader(meta_by_worker[i], read_data_q, i) for i in range(n_workers)]
    reader_procs = [Process(target=r.work, args=()) for r in readers]
    print('Starting worker processes...')
    for reader_proc in reader_procs:
        reader_proc.start()

    form_embds = {}
    batting_order_ids = {}
    n_term_rcvd = 0
    while True:
        if read_data_q.empty():
            print('Orchestrator sleeping...')
            time.sleep(10)
        else:
            in_data = read_data_q.get()
            if in_data == '[TERM]':
                n_term_rcvd += 1
                print('Orchestrator received {} term signals...'.format(n_term_rcvd))

                if n_term_rcvd == n_workers:
                    break
            else:
                player_id, form_embd, batting_order = in_data

                curr_player_form_embds = form_embds.get(player_id, [])
                curr_player_form_embds.append(form_embd)
                form_embds[player_id] = curr_player_form_embds

                curr_player_batting_order = batting_order_ids.get(player_id, [])
                curr_player_batting_order.append(batting_order)
                batting_order_ids[player_id] = curr_player_batting_order

    for player_id in form_embds.keys():
        form_embd_list = form_embds[player_id]
        form_embd_mat = np.vstack(form_embd_list)
        avg_form_embds = np.mean(form_embd_mat, axis=0)
        form_embds[player_id] = [float(i) for i in list(avg_form_embds)]

        batting_order_vals = batting_order_ids[player_id]
        avg_batting_order = sum(batting_order_vals) / len(batting_order_vals)
        batting_order_ids[player_id] = avg_batting_order

    return form_embds, batting_order_ids


def parse_json(j, j_type, max_value_data=None, data_scopes_to_use=None):
    if j_type == 'pitcher':
        avoid_keys = ['__id__', 'throws', 'first_name', 'last_name', '__bio__']
    elif j_type == 'batter':
        avoid_keys = ['__id__', 'stand', 'first_name', 'last_name', '__bio__']
    else:
        avoid_keys = []

    if max_value_data is not None:
        # print('MAX VALUE DATA BEING USED')
        type_max_value_data = max_value_data[j_type]
    else:
        type_max_value_data = {}

    data = []
    for k1, v1 in j.items():
        k1_norm_vals = type_max_value_data.get(k1, {})
        if k1 not in avoid_keys and (data_scopes_to_use is None or k1 in data_scopes_to_use):
            for k2, v2 in v1.items():
                if type(v2) == list:
                    k2_norm_val = k1_norm_vals.get(k2, [1.0])
                    data.extend(
                        [float(item_val) / norm_val if not math.isnan(item_val) and not item_val == math.inf else 1.0
                         for item_val, norm_val in zip(v2, k2_norm_val)])
                    # pass
                else:
                    if v2 == math.inf:
                        new_v2 = 1.0
                    elif math.isnan(v2):
                        new_v2 = 0.0
                    else:
                        k2_norm_val = k1_norm_vals.get(k2, 1.0)
                        new_v2 = v2 / k2_norm_val if k2_norm_val != 0 else v2
                        # print('k1: {} k2: {} raw v2: {} max val: {} new v2: {}'.format(k1,
                        #                                                                k2,
                        #                                                                v2,
                        #                                                                k2_norm_val,
                        #                                                                new_v2))
                    data.append(float(new_v2))
    if math.nan in data:
        input('j: {}'.format(j))

    return data


def read_starter_stats(whole_record_dir, norm_vals, player_type='batter', data_scopes_to_use=None):
    intermediate_stats = {}

    # for season in os.listdir(whole_record_dir):
    for season in ['2015', '2016', '2017', '2018', '2019']:
        print('Reading data for {} season...'.format(season))
        season_dir = os.path.join(whole_record_dir, season)

        for game_filename in os.listdir(season_dir):
            game_fp = os.path.join(season_dir, game_filename)
            game_pk = int(game_filename[:-5])
            game_j = json.load(open(game_fp))
            if player_type == 'batter':
                for home_batter_id in game_j['home_batting_order']:
                    batter_j = game_j['home_batters'][str(home_batter_id)]
                    # print('batter_j: {}'.format(batter_j))
                    batter_stats = parse_json(batter_j, j_type='batter', max_value_data=norm_vals,
                                              data_scopes_to_use=data_scopes_to_use)
                    curr_player_stats = intermediate_stats.get(home_batter_id, [])
                    curr_player_stats.append(batter_stats)
                    intermediate_stats[home_batter_id] = curr_player_stats

                for away_batter_id in game_j['away_batting_order']:
                    batter_j = game_j['away_batters'][str(away_batter_id)]
                    batter_stats = parse_json(batter_j, j_type='batter', max_value_data=norm_vals,
                                              data_scopes_to_use=data_scopes_to_use)
                    curr_player_stats = intermediate_stats.get(away_batter_id, [])
                    curr_player_stats.append(batter_stats)
                    intermediate_stats[away_batter_id] = curr_player_stats
            else:
                home_sp_j = game_j['home_starter']
                home_sp_id = home_sp_j['__id__']
                home_sp_stats = parse_json(home_sp_j, j_type='pitcher', max_value_data=norm_vals,
                                           data_scopes_to_use=data_scopes_to_use)
                curr_player_stats = intermediate_stats.get(home_sp_id, [])
                curr_player_stats.append(home_sp_stats)
                intermediate_stats[home_sp_id] = curr_player_stats

                away_sp_j = game_j['away_starter']
                away_sp_id = away_sp_j['__id__']
                away_sp_stats = parse_json(away_sp_j, j_type='pitcher', max_value_data=norm_vals,
                                           data_scopes_to_use=data_scopes_to_use)
                curr_player_stats = intermediate_stats.get(away_sp_id, [])
                curr_player_stats.append(away_sp_stats)
                intermediate_stats[away_sp_id] = curr_player_stats

    final_player_stats = {}
    for k, v in intermediate_stats.items():
        v = np.array(v)
        v = list(np.mean(v, axis=0).reshape(-1))
        final_player_stats[str(k)] = v

    return final_player_stats


def query_db(db_fp, query, args=None):
    conn = sqlite3.connect(db_fp, check_same_thread=False)
    c = conn.cursor()
    if args is None:
        c.execute(query)
    else:
        c.execute(query, args)
    rows = c.fetchall()

    return rows


def parse_salary(x):
    x = x.strip()
    if x[0] == '(':
        neg = True
    else:
        neg = False

    num_strs = [str(i) for i in range(10)]

    while x[0] not in num_strs:
        x = x[1:]

    while x[-1] not in num_strs:
        x = x[:-1]

    x = float(x)
    if neg:
        x = x * -1.0

    return x


def read_salary_war_pos_hand(player_ids, db_fp, bio_info_d, player_type='batter'):
    player_id_to_salary = {}
    player_id_to_war = {}
    player_id_to_pos = {}
    player_id_to_hand = {}

    if player_type == 'batter':
        salary_war_query = """select Dol, WAR
                                from batting_by_season
                                where Name like ? and Season >= 2015"""

        handedness_query = """select distinct(stand)
                                      from statcast
                                      where batter = ?"""
    else:
        salary_war_query = """select Dollars, WAR
                                        from pitching_by_season
                                        where Name like ? and Season >= 2015"""

        handedness_query = """select distinct(p_throws)
                                              from statcast
                                              where pitcher = ?"""

    for player_id in player_ids:
        player_pos = bio_info_d[str(player_id)]['mlb_pos' if player_type == 'batter' else 'cbs_pos']
        player_fullname = '{} {}'.format(bio_info_d[str(player_id)]['name_first'],
                                         bio_info_d[str(player_id)]['name_last'])
        # print('player_fullname: {}'.format(player_fullname))

        salary_war_args = (player_fullname,)
        salary_war_res = query_db(db_fp, salary_war_query, salary_war_args)
        # print('player_pos: {}'.format(player_pos))
        # print('salary_war_res: {}'.format(salary_war_res))
        player_id_to_pos[player_id] = player_pos

        handedness_args = (player_id,)
        player_handedness_res = query_db(db_fp, handedness_query, handedness_args)

        if len(player_handedness_res) > 1:
            player_id_to_hand[player_id] = 'S'
        else:
            player_id_to_hand[player_id] = player_handedness_res[0][0]

        # input('player_handedness_res: {}'.format(player_handedness_res))

        if len(salary_war_res) > 0:
            all_player_salary, all_player_war = map(list, zip(*salary_war_res))
            # print('all_player_salary: {}'.format(all_player_salary))
            # print('all_player_war: {}'.format(all_player_war))
            all_player_salary = [parse_salary(sal) for sal in all_player_salary]

            avg_salary = sum(all_player_salary) / len(all_player_salary)
            avg_war = sum(all_player_war) / len(all_player_war)

            player_id_to_salary[player_id] = avg_salary
            player_id_to_war[player_id] = avg_war

            # input('all_player_salary: {} all_player_sall_player_waralary: {}'.format(all_player_salary, all_player_war))

    print('len(player_id_to_salary): {}'.format(len(player_id_to_salary)))
    print('len(player_id_to_war): {}'.format(len(player_id_to_war)))
    print('len(player_id_to_pos): {}'.format(len(player_id_to_pos)))
    print('len(player_id_to_hand): {}'.format(len(player_id_to_hand)))

    return player_id_to_salary, player_id_to_war, player_id_to_pos, player_id_to_hand


def calc_batting_avg_from_event_d(event_d):
    if event_d.get('home_run', 0) + event_d.get('single', 0) + event_d.get('double', 0) + event_d.get('triple', 0) == 0:
        batting_avg = 0.0
    else:
        batting_avg = (event_d.get('home_run', 0) + event_d.get('single', 0) + event_d.get('double', 0) +
                       event_d.get('triple', 0)) / (event_d.get('home_run', 0) + event_d.get('single', 0) +
                                                    event_d.get('double', 0) + event_d.get('triple', 0) +
                                                    event_d.get('strikeout', 0) + event_d.get('field_out', 0) +
                                                    event_d.get('force_out', 0) +
                                                    event_d.get('grounded_into_double_play', 0) +
                                                    event_d.get('strikeout_double_play', 0) +
                                                    event_d.get('double_play', 0) + event_d.get('triple_play', 0))

    return batting_avg


def parse_batting_stats(pa_df):
    batting_avg = 0.0
    batting_avg_v_lhp = None
    batting_avg_v_rhp = None

    all_events = pa_df['events'].tolist()
    u_events, u_event_counts = np.unique(all_events, return_counts=True)
    all_event_d = dict(zip(u_events, u_event_counts))
    batting_avg = calc_batting_avg_from_event_d(all_event_d)

    events_v_lhp = pa_df[pa_df['p_throws'] == 'L']['events'].tolist()
    if len(events_v_lhp) > 0:
        u_events_v_lhp, u_event_counts_v_lhp = np.unique(events_v_lhp, return_counts=True)
        all_event_v_lhp_d = dict(zip(u_events_v_lhp, u_event_counts_v_lhp))
        batting_avg_v_lhp = calc_batting_avg_from_event_d(all_event_v_lhp_d)

    events_v_rhp = pa_df[pa_df['p_throws'] == 'R']['events'].tolist()
    if len(events_v_rhp) > 0:
        u_events_v_rhp, u_event_counts_v_rhp = np.unique(events_v_rhp, return_counts=True)
        all_event_v_rhp_d = dict(zip(u_events_v_rhp, u_event_counts_v_rhp))
        batting_avg_v_rhp = calc_batting_avg_from_event_d(all_event_v_rhp_d)

    return batting_avg, batting_avg_v_lhp, batting_avg_v_rhp


def calc_player_batting_stats(player_ids, db_fp, player_bio_info):
    batting_stats = {}
    pa_query = """select game_pk, at_bat_number, pitch_number, events, p_throws, 
                    hit_distance_sc, launch_speed, launch_angle, type, hc_x, hc_y
                  from statcast
                  where game_year >= 2015 and game_year <= 2019 and batter = ?"""

    season_stats_query = """select sum(AB), sum(PA), sum(H), sum(B1), sum(B2), sum(B3), sum(HR), sum(RBI), sum(BB),
                                sum(SO)
                            from batting_by_season
                            where Season >= 2015 and Season <= 2019 and Name = ?"""

    event_map_hit = {
          "single": 1,
          "double": 1,
          "triple": 1,
          "home_run": 1,
          "field_out": 0,
          "strikeout": 0,
          "walk": 0,
          "intent_walk": 0,
          "hit_by_pitch": 0,
          "sac_fly": 0,
          "sac_bunt": 0,
          "double_play": 0,
          "triple_play": 0,
          "fielders_choice": 0,
    }

    for player_id in player_ids:
        pa_args = (player_id,)
        pa_res = query_db(db_fp, pa_query, pa_args)
        # pa_events, pa_p_throws = map(list, zip(*pa_res))
        pa_df = pd.DataFrame(pa_res, columns=['game_pk', 'at_bat_number', 'pitch_number', 'events', 'p_throws',
                                              'hit_distance_sc', 'launch_speed', 'launch_angle', 'type',
                                              'hc_x', 'hc_y'])

        player_game_pks = pa_df['game_pk'].tolist()
        player_ab_nos = pa_df['at_bat_number'].tolist()
        player_pa_ids = ['{}-{}'.format(g_pk, ab_no) for g_pk, ab_no in zip(player_game_pks, player_ab_nos)]
        u_player_pa_ids = list(np.unique(player_pa_ids))
        # input('u_player_pa_ids: {}'.format(u_player_pa_ids))
        pa_df = pa_df[pa_df['events'].notnull()]
        avg, avg_v_lhp, avg_v_rhp = parse_batting_stats(pa_df)

        pa_df = pa_df[pa_df['pitch_number'] == 1]
        p_throws = pa_df['p_throws'].tolist()
        n_rhp = [1 if pt == 'R' else 0 for pt in p_throws]
        pct_rhp = sum(n_rhp) / len(n_rhp) if len(n_rhp) > 0 else None

        pa_df['is_hit'] = pa_df['events']
        pa_df['is_hit'] = pa_df['is_hit'].map(event_map_hit)

        bip_df = pa_df[pa_df['type'] == 'X']
        bip_df = bip_df[bip_df['is_hit'] == 1]

        hit_distance_sc = bip_df['hit_distance_sc'].tolist()
        launch_speed = bip_df['launch_speed'].tolist()
        launch_angle = bip_df['launch_angle'].tolist()
        hc_x = bip_df['hc_x'].tolist()
        hc_y = bip_df['hc_y'].tolist()

        hit_distance_sc = [x for x in hit_distance_sc if str(x) != 'nan' and x is not None]
        launch_speed = [x for x in launch_speed if str(x) != 'nan' and x is not None]
        launch_angle = [x for x in launch_angle if str(x) != 'nan' and x is not None]
        hc_x = [x for x in hc_x if str(x) != 'nan' and x is not None]
        hc_y = [x for x in hc_y if str(x) != 'nan' and x is not None]

        avg_hit_distance = sum(hit_distance_sc) / len(hit_distance_sc) if len(hit_distance_sc) > 0 else -1
        avg_launch_speed = sum(launch_speed) / len(launch_speed) if len(launch_speed) > 0 else -1
        avg_launch_angle = sum(launch_angle) / len(launch_angle) if len(launch_angle) > 0 else -1
        avg_hc_x = sum(hc_x) / len(hc_x) if len(hc_x) > 0 else -1
        avg_hc_y = sum(hc_y) / len(hc_y) if len(hc_y) > 0 else -1

        player_db_name = player_bio_info[str(player_id)]['fangraphs_name']
        season_stats_args = (player_db_name,)
        season_stats_res = query_db(db_fp, season_stats_query, season_stats_args)[0]
        if season_stats_res[1] is not None and season_stats_res[1] > 0:
            single_rate = season_stats_res[3] / season_stats_res[1]
            double_rate = season_stats_res[4] / season_stats_res[1]
            triple_rate = season_stats_res[5] / season_stats_res[1]
            hr_rate = season_stats_res[6] / season_stats_res[1]
            rbi_rate = season_stats_res[7] / season_stats_res[1]
            walk_rate = season_stats_res[8] / season_stats_res[1]
            strikeout_rate = season_stats_res[9] / season_stats_res[1]
        else:
            single_rate = -1
            double_rate = -1
            triple_rate = -1
            hr_rate = -1
            rbi_rate = -1
            walk_rate = -1
            strikeout_rate = -1

        batting_stats[player_id] = {'avg': avg, 'avg_v_lhp': avg_v_lhp, 'avg_v_rhp': avg_v_rhp,
                                    'n_pa': len(u_player_pa_ids), 'pct_rhp': pct_rhp,
                                    'avg_hit_distance': avg_hit_distance, 'avg_launch_speed': avg_launch_speed,
                                    'avg_launch_angle': avg_launch_angle, 'avg_hc_x': avg_hc_x, 'avg_hc_y': avg_hc_y,
                                    'single_rate': single_rate, 'double_rate': double_rate, 'triple_rate': triple_rate,
                                    'hr_rate': hr_rate, 'rbi_rate': rbi_rate, 'walk_rate': walk_rate,
                                    'strikeout_rate': strikeout_rate}

    return batting_stats


def do_pca(x, n=2):
    pca_model = PCA(n_components=n)
    new_x = pca_model.fit_transform(x)

    return new_x


def do_tsne(x, perplexity, n_iter):
    tsne = TSNE(2, verbose=1, n_jobs=8, perplexity=perplexity, n_iter=n_iter)
    tsne_proj = tsne.fit_transform(x)
    print('\ttsne_proj: {}'.format(tsne_proj.shape))

    return tsne_proj


def quick_map(pos_str):
    pos_str = pos_str.strip()

    if pos_str == 'UNK':
        pos_str = 'UNK'
    elif pos_str == 'P':
        pos_str = 'Pitcher'
    elif pos_str == 'C':
        pos_str = 'Catcher'
    elif pos_str in ['1B', '2B', '3B', 'SS']:
        pos_str = 'Infield'
    elif pos_str in ['CF', 'LF', 'OF', 'RF']:
        pos_str = 'Outfield'
    elif pos_str in ['DH']:
        pos_str = 'DH'

    return pos_str


def make_batter_plots(xs, ys, avg_batting_order_idx, ordered_player_salaries, ordered_player_wars, ordered_player_pos,
                      ordered_player_handedness, batting_avg, batting_avg_v_lhp, batting_avg_v_rhp, n_pa, pct_rhp,
                      avg_hit_distance, avg_launch_speed, avg_launch_angle, avg_hc_x, avg_hc_y, allstar_apps,
                      ordered_single_rate, ordered_double_rate, ordered_triple_rate, ordered_hr_rate, ordered_rbi_rate,
                      ordered_walk_rate, ordered_strikeout_rate, ordered_hr_crowns, ordered_batting_titles,
                      ordered_rbi_titles, ordered_stolen_base_titles, plt_fp):
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'chartreuse', 'pink', 'yellow', 'black',
              'skyblue', 'red', 'lavender', 'olive']
    n_cols = 6
    n_rows = 5

    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
    # cmap = plt.cm.cool
    # cmap = plt.cm.Blues
    # cmap = plt.cm.hot
    # cmap = plt.cm.spring
    # cmap = plt.cm.YlOrRd
    cmap = plt.cm.YlGnBu

    print('Plotting batting order...')
    plt.subplot(n_rows, n_cols, 1)
    plt.scatter(xs, ys, c=avg_batting_order_idx, cmap=cmap, alpha=0.5)
    plt.title('Batting Order')
    plt.colorbar()

    print('Plotting salary...')
    plt.subplot(n_rows, n_cols, 2)
    salary_xs = [xs[i] for i in range(len(ordered_player_salaries)) if ordered_player_salaries[i] is not None]
    salary_ys = [ys[i] for i in range(len(ordered_player_salaries)) if ordered_player_salaries[i] is not None]
    ordered_player_salaries = [ops for ops in ordered_player_salaries if ops is not None]
    plt.scatter(salary_xs, salary_ys, c=ordered_player_salaries, cmap=cmap, alpha=0.5)
    plt.title('Salary')
    plt.colorbar()

    print('Plotting WAR...')
    plt.subplot(n_rows, n_cols, 3)
    war_xs = [xs[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and 'P' not in ordered_player_pos[i]]
    war_ys = [ys[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and 'P' not in ordered_player_pos[i]]
    ordered_player_wars = [opw for i, opw in enumerate(ordered_player_wars) if
                           opw is not None and 'P' not in ordered_player_pos[i]]
    plt.scatter(war_xs, war_ys, c=ordered_player_wars, cmap=cmap, alpha=0.5)
    plt.title('War')
    plt.colorbar()

    print('Plotting position 1...')
    plt.subplot(n_rows, n_cols, 4)
    unique_player_positions = list(np.unique(ordered_player_pos))
    unique_player_positions = [upp for upp in unique_player_positions if upp != 'UNK']
    print('unique_player_positions: {}'.format(unique_player_positions))
    for pos_idx, pos_label in enumerate(unique_player_positions):
        pos_xs = [xs[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
        pos_ys = [ys[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
        pos_color = colors[pos_idx]
        plt.scatter(pos_xs, pos_ys, c=pos_color, label=pos_label, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Position (Raw)')

    ordered_player_pos = [quick_map(opp) for opp in ordered_player_pos]
    print('Plotting position 2...')
    plt.subplot(n_rows, n_cols, 5)
    unique_player_positions = list(np.unique(ordered_player_pos))
    unique_player_positions = [upp for upp in unique_player_positions if upp != 'UNK']
    print('unique_player_positions: {}'.format(unique_player_positions))
    for pos_idx, pos_label in enumerate(unique_player_positions):
        pos_xs = [xs[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
        pos_ys = [ys[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
        pos_color = colors[pos_idx]
        plt.scatter(pos_xs, pos_ys, c=pos_color, label=pos_label, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Position (Rough)')

    print('Plotting handedness...')
    plt.subplot(n_rows, n_cols, 6)
    for hand_idx, hand_label in enumerate(['R', 'L', 'S']):
        hand_xs = [xs[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_ys = [ys[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_color = colors[hand_idx]
        plt.scatter(hand_xs, hand_ys, c=hand_color, label=hand_label, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Handedness')

    print('Plotting batting avg...')
    plt.subplot(n_rows, n_cols, 7)
    plt.scatter(xs, ys, c=batting_avg, cmap=cmap, alpha=0.5)
    plt.title('Batting Avg')
    plt.colorbar()

    xs_lhp = [xs[i] for i, _ in enumerate(batting_avg_v_lhp) if
              batting_avg_v_lhp[i] is not None and batting_avg_v_lhp[i] < 0.95]
    ys_lhp = [ys[i] for i, _ in enumerate(batting_avg_v_lhp) if
              batting_avg_v_lhp[i] is not None and batting_avg_v_lhp[i] < 0.95]
    batting_avg_v_lhp = [bavl for bavl in batting_avg_v_lhp if bavl is not None and bavl < 0.95]
    plt.subplot(n_rows, n_cols, 8)
    plt.scatter(xs_lhp, ys_lhp, c=batting_avg_v_lhp, cmap=cmap, alpha=0.5)
    plt.title('Batting Avg v LHP')
    plt.colorbar()

    xs_rhp = [xs[i] for i, _ in enumerate(batting_avg_v_rhp) if batting_avg_v_rhp[i] is not None]
    ys_rhp = [ys[i] for i, _ in enumerate(batting_avg_v_rhp) if batting_avg_v_rhp[i] is not None]
    batting_avg_v_rhp = [bavr for bavr in batting_avg_v_rhp if bavr is not None]
    plt.subplot(n_rows, n_cols, 9)
    plt.scatter(xs_rhp, ys_rhp, c=batting_avg_v_rhp, cmap=cmap, alpha=0.5)
    plt.title('Batting Avg v RHP')
    plt.colorbar()

    print('Plotting n PA...')
    plt.subplot(n_rows, n_cols, 10)
    plt.scatter(xs, ys, c=n_pa, cmap=cmap, alpha=0.5)
    plt.title('# Plate Appearances')
    plt.colorbar()

    rhp_xs = [xs[i] for i in range(len(pct_rhp)) if pct_rhp[i] is not None]
    rhp_ys = [ys[i] for i in range(len(pct_rhp)) if pct_rhp[i] is not None]
    pct_rhp = [prhp for prhp in pct_rhp if prhp is not None]
    plt.subplot(n_rows, n_cols, 11)
    plt.scatter(rhp_xs, rhp_ys, c=pct_rhp, cmap=cmap, alpha=0.5)
    plt.title('Pct PA vs RHP')
    plt.colorbar()

    pct_lhp = [1.0 - prhp for prhp in pct_rhp]
    plt.subplot(n_rows, n_cols, 12)
    plt.scatter(rhp_xs, rhp_ys, c=pct_lhp, cmap=cmap, alpha=0.5)
    plt.title('Pct PA vs LHP')
    plt.colorbar()

    print('Plotting hit dist...')
    plt.subplot(n_rows, n_cols, 13)
    plt.scatter(xs, ys, c=avg_hit_distance, cmap=cmap, alpha=0.5)
    plt.title('Avg. Hit Dist.')
    plt.colorbar()

    print('Plotting launch speed...')
    plt.subplot(n_rows, n_cols, 14)
    plt.scatter(xs, ys, c=avg_launch_speed, cmap=cmap, alpha=0.5)
    plt.title('Avg. Launch Speed')
    plt.colorbar()

    print('Plotting launch angle...')
    plt.subplot(n_rows, n_cols, 15)
    plt.scatter(xs, ys, c=avg_launch_angle, cmap=cmap, alpha=0.5)
    plt.title('Avg. Launch Angle')
    plt.colorbar()

    print('Plotting hc x...')
    plt.subplot(n_rows, n_cols, 16)
    plt.scatter(xs, ys, c=avg_hc_x, cmap=cmap, alpha=0.5)
    plt.title('Avg. Hit Coord X')
    plt.colorbar()

    print('Plotting hc y...')
    plt.subplot(n_rows, n_cols, 17)
    plt.scatter(xs, ys, c=avg_hc_y, cmap=cmap, alpha=0.5)
    plt.title('Avg. Hit Coord Y')
    plt.colorbar()

    print('Plotting all star apps...')
    plt.subplot(n_rows, n_cols, 18)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        as_ys = [ys[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        # as_color = colors[as_idx] if as_idx != 0 else 'lightgrey'
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]

        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('All-star Apps')

    plt.subplot(n_rows, n_cols, 19)
    plt.scatter(xs, ys, c=allstar_apps, cmap=cmap, alpha=0.5)
    plt.title('All-star Apps')
    plt.colorbar()

    single_xs = [xs[i] for i in range(len(ordered_single_rate)) if ordered_single_rate[i] >= 0.0]
    single_ys = [ys[i] for i in range(len(ordered_single_rate)) if ordered_single_rate[i] >= 0.0]
    ordered_single_rate = [osr for osr in ordered_single_rate if osr >= 0.0]
    plt.subplot(n_rows, n_cols, 20)
    plt.scatter(single_xs, single_ys, c=ordered_single_rate, cmap=cmap, alpha=0.5)
    plt.title('Single Rate')
    plt.colorbar()

    double_xs = [xs[i] for i in range(len(ordered_double_rate)) if ordered_double_rate[i] >= 0.0]
    double_ys = [ys[i] for i in range(len(ordered_double_rate)) if ordered_double_rate[i] >= 0.0]
    ordered_double_rate = [v for v in ordered_double_rate if v >= 0.0]
    plt.subplot(n_rows, n_cols, 21)
    plt.scatter(double_xs, double_ys, c=ordered_double_rate, cmap=cmap, alpha=0.5)
    plt.title('Double Rate')
    plt.colorbar()

    triple_xs = [xs[i] for i in range(len(ordered_triple_rate)) if ordered_triple_rate[i] >= 0.0]
    triple_ys = [ys[i] for i in range(len(ordered_triple_rate)) if ordered_triple_rate[i] >= 0.0]
    ordered_triple_rate = [v for v in ordered_triple_rate if v >= 0.0]
    plt.subplot(n_rows, n_cols, 22)
    plt.scatter(triple_xs, triple_ys, c=ordered_triple_rate, cmap=cmap, alpha=0.5)
    plt.title('Triple Rate')
    plt.colorbar()

    hr_xs = [xs[i] for i in range(len(ordered_hr_rate)) if ordered_hr_rate[i] >= 0.0]
    hr_ys = [ys[i] for i in range(len(ordered_hr_rate)) if ordered_hr_rate[i] >= 0.0]
    ordered_hr_rate = [v for v in ordered_hr_rate if v >= 0.0]
    plt.subplot(n_rows, n_cols, 23)
    plt.scatter(hr_xs, hr_ys, c=ordered_hr_rate, cmap=cmap, alpha=0.5)
    plt.title('HR Rate')
    plt.colorbar()

    rbi_xs = [xs[i] for i in range(len(ordered_rbi_rate)) if ordered_rbi_rate[i] >= 0.0]
    rbi_ys = [ys[i] for i in range(len(ordered_rbi_rate)) if ordered_rbi_rate[i] >= 0.0]
    ordered_rbi_rate = [v for v in ordered_rbi_rate if v >= 0.0]
    plt.subplot(n_rows, n_cols, 24)
    plt.scatter(rbi_xs, rbi_ys, c=ordered_rbi_rate, cmap=cmap, alpha=0.5)
    plt.title('RBI/PA')
    plt.colorbar()

    walk_xs = [xs[i] for i in range(len(ordered_walk_rate)) if ordered_walk_rate[i] >= 0.0]
    walk_ys = [ys[i] for i in range(len(ordered_walk_rate)) if ordered_walk_rate[i] >= 0.0]
    ordered_walk_rate = [v for v in ordered_walk_rate if v >= 0.0]
    plt.subplot(n_rows, n_cols, 25)
    plt.scatter(walk_xs, walk_ys, c=ordered_walk_rate, cmap=cmap, alpha=0.5)
    plt.title('Walk Rate')
    plt.colorbar()

    strikeout_xs = [xs[i] for i in range(len(ordered_strikeout_rate)) if 0.9 >= ordered_strikeout_rate[i] >= 0.0]
    strikeout_ys = [ys[i] for i in range(len(ordered_strikeout_rate)) if 0.9 >= ordered_strikeout_rate[i] >= 0.0]
    ordered_strikeout_rate = [v for v in ordered_strikeout_rate if 0.9 >= v >= 0.0]
    plt.subplot(n_rows, n_cols, 26)
    plt.scatter(strikeout_xs, strikeout_ys, c=ordered_strikeout_rate, cmap=cmap, alpha=0.5)
    plt.title('Strikeout Rate')
    plt.colorbar()

    plt.subplot(n_rows, n_cols, 27)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(ordered_hr_crowns)) if ordered_hr_crowns[i] == as_label]
        as_ys = [ys[i] for i in range(len(ordered_hr_crowns)) if ordered_hr_crowns[i] == as_label]
        # as_color = colors[as_idx] if as_idx != 0 else 'lightgrey'
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]

        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('HR Crowns')

    plt.subplot(n_rows, n_cols, 28)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(ordered_batting_titles)) if ordered_batting_titles[i] == as_label]
        as_ys = [ys[i] for i in range(len(ordered_batting_titles)) if ordered_batting_titles[i] == as_label]
        # as_color = colors[as_idx] if as_idx != 0 else 'lightgrey'
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]

        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Batting Titles')

    plt.subplot(n_rows, n_cols, 29)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(ordered_rbi_titles)) if ordered_rbi_titles[i] == as_label]
        as_ys = [ys[i] for i in range(len(ordered_rbi_titles)) if ordered_rbi_titles[i] == as_label]
        # as_color = colors[as_idx] if as_idx != 0 else 'lightgrey'
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]

        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('RBI Titles')

    plt.subplot(n_rows, n_cols, 30)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(ordered_stolen_base_titles)) if ordered_stolen_base_titles[i] == as_label]
        as_ys = [ys[i] for i in range(len(ordered_stolen_base_titles)) if ordered_stolen_base_titles[i] == as_label]
        # as_color = colors[as_idx] if as_idx != 0 else 'lightgrey'
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]

        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Stolen Base Titles')

    plt.suptitle('Batter Form tSNE')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.2)
    plt.savefig(plt_fp, bbox_inches='tight')
    plt.clf()


def make_succinct_batter_plots(xs, ys, allstar_apps, ordered_player_wars, ordered_hr_rate,
                               ordered_player_handedness, ordered_player_pos, plt_fp):
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'chartreuse', 'pink', 'yellow', 'black',
              'skyblue', 'red', 'lavender', 'olive']
    cmap = plt.cm.YlGnBu
    n_cols = 4
    n_rows = 1

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))

    print('Plotting all star apps...')
    plt.subplot(n_rows, n_cols, 1)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        as_ys = [ys[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        # as_color = colors[as_idx] if as_idx != 0 else 'lightgrey'
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]

        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('All-star Apps')

    print('Plotting WAR...')
    plt.subplot(n_rows, n_cols, 2)
    war_xs = [xs[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and 'P' not in ordered_player_pos[i]]
    war_ys = [ys[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and 'P' not in ordered_player_pos[i]]
    ordered_player_wars = [opw for i, opw in enumerate(ordered_player_wars) if
                           opw is not None and 'P' not in ordered_player_pos[i]]
    plt.scatter(war_xs, war_ys, c=ordered_player_wars, cmap=cmap, alpha=0.5)
    plt.title('War')
    plt.colorbar()

    hr_xs = [xs[i] for i in range(len(ordered_hr_rate)) if ordered_hr_rate[i] >= 0.0]
    hr_ys = [ys[i] for i in range(len(ordered_hr_rate)) if ordered_hr_rate[i] >= 0.0]
    ordered_hr_rate = [v for v in ordered_hr_rate if v >= 0.0]
    plt.subplot(n_rows, n_cols, 3)
    plt.scatter(hr_xs, hr_ys, c=ordered_hr_rate, cmap=cmap, alpha=0.5)
    plt.title('HR Rate')
    plt.colorbar()

    print('Plotting handedness...')
    plt.subplot(n_rows, n_cols, 4)
    for hand_idx, hand_label in enumerate(['R', 'L', 'S']):
        hand_xs = [xs[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_ys = [ys[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_color = colors[hand_idx]
        plt.scatter(hand_xs, hand_ys, c=hand_color, label=hand_label, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Handedness')

    plt.suptitle('Batter Form tSNE', y=1.0)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.2)
    plt.savefig(plt_fp, bbox_inches='tight')
    plt.clf()


def inspect_batter_form_embeddings(args, form_desc, all_form_embds, batting_order_ids, player_bio_info):
    player_salary_fp = os.path.join(bin_dir, 'player_salary_{}.json'.format(form_desc))
    player_war_fp = os.path.join(bin_dir, 'player_war_{}.json'.format(form_desc))
    player_pos_fp = os.path.join(bin_dir, 'player_pos_{}.json'.format(form_desc))
    player_handedness_fp = os.path.join(bin_dir, 'player_handedness_{}.json'.format(form_desc))
    player_batting_stats_fp = os.path.join(bin_dir, 'player_batting_stats_{}.json'.format(form_desc))
    ordered_batter_ids_fp = os.path.join(bin_dir, 'ordered_batter_ids_{}.json'.format(form_desc))

    if os.path.exists(player_salary_fp) and os.path.exists(player_war_fp) and os.path.exists(player_pos_fp) \
            and os.path.exists(player_handedness_fp):
        print('Reading meta data...')
        player_salary = json.load(open(player_salary_fp))
        player_war = json.load(open(player_war_fp))
        player_pos = json.load(open(player_pos_fp))
        player_handedness = json.load(open(player_handedness_fp))
    else:
        print('Computing meta data...')
        all_player_ids = list(all_form_embds.keys())
        player_salary, player_war, player_pos, player_handedness = read_salary_war_pos_hand(all_player_ids,
                                                                                            args.db_fp, player_bio_info,
                                                                                            player_type='batter')
        with open(player_salary_fp, 'w+') as f:
            f.write(json.dumps(player_salary, indent=1))

        with open(player_war_fp, 'w+') as f:
            f.write(json.dumps(player_war, indent=1))

        with open(player_pos_fp, 'w+') as f:
            f.write(json.dumps(player_pos, indent=1))

        with open(player_handedness_fp, 'w+') as f:
            f.write(json.dumps(player_handedness, indent=1))

    all_player_ids = list(all_form_embds.keys())
    if os.path.exists(player_batting_stats_fp):
        print('Reading player batting stats...')
        player_batting_stats = json.load(open(player_batting_stats_fp))
    else:
        print('Calculating player batting stats...')
        player_batting_stats = calc_player_batting_stats(all_player_ids, args.db_fp, player_bio_info)

        with open(player_batting_stats_fp, 'w+') as f:
            f.write(json.dumps(player_batting_stats, indent=1))

    print('Subsetting player IDs w/ >= {} PAs...'.format(args.min_pa))
    all_player_ids = [api for api in all_player_ids if player_batting_stats.get(api, {}).get('n_pa', 0) >= args.min_pa]
    with open(ordered_batter_ids_fp, 'w+') as f:
        f.write(json.dumps(all_player_ids, indent=1))

    all_player_embds = np.vstack([all_form_embds[p_id] for p_id in all_player_ids])
    avg_batting_order_idx = [batting_order_ids[p_id] for p_id in all_player_ids]
    ordered_player_salaries = [player_salary.get(p_id, None) for p_id in all_player_ids]
    ordered_player_wars = [player_war.get(p_id, None) for p_id in all_player_ids]
    # ordered_player_pos = [quick_map(player_pos[p_id]) for p_id in all_player_ids]
    ordered_player_pos = [player_pos[p_id] for p_id in all_player_ids]
    ordered_player_handedness = [player_handedness.get(p_id, None) for p_id in all_player_ids]
    ordered_batting_avg = [player_batting_stats.get(p_id, {}).get('avg', None) for p_id in all_player_ids]
    ordered_batting_avg_v_lhp = [player_batting_stats.get(p_id, {}).get('avg_v_lhp', None) for p_id in all_player_ids]
    ordered_batting_avg_v_rhp = [player_batting_stats.get(p_id, {}).get('avg_v_rhp', None) for p_id in all_player_ids]
    ordered_n_pa = [player_batting_stats.get(p_id, {}).get('n_pa', None) for p_id in all_player_ids]
    ordered_pct_rhp = [player_batting_stats.get(p_id, {}).get('pct_rhp', None) for p_id in all_player_ids]
    ordered_avg_hit_distance = [player_batting_stats.get(p_id, {}).get('avg_hit_distance', -1) for p_id in
                                all_player_ids]
    ordered_avg_launch_speed = [player_batting_stats.get(p_id, {}).get('avg_launch_speed', -1) for p_id in
                                all_player_ids]
    ordered_avg_launch_angle = [player_batting_stats.get(p_id, {}).get('avg_launch_angle', -1) for p_id in
                                all_player_ids]
    ordered_avg_hc_x = [player_batting_stats.get(p_id, {}).get('avg_hc_x', -1) for p_id in all_player_ids]
    ordered_avg_hc_y = [player_batting_stats.get(p_id, {}).get('avg_hc_y', -1) for p_id in all_player_ids]
    ordered_single_rate = [player_batting_stats.get(p_id, {}).get('single_rate', -1) for p_id in all_player_ids]
    ordered_double_rate = [player_batting_stats.get(p_id, {}).get('double_rate', -1) for p_id in all_player_ids]
    ordered_triple_rate = [player_batting_stats.get(p_id, {}).get('triple_rate', -1) for p_id in all_player_ids]
    ordered_hr_rate = [player_batting_stats.get(p_id, {}).get('hr_rate', -1) for p_id in all_player_ids]
    ordered_rbi_rate = [player_batting_stats.get(p_id, {}).get('rbi_rate', -1) for p_id in all_player_ids]
    ordered_walk_rate = [player_batting_stats.get(p_id, {}).get('walk_rate', -1) for p_id in all_player_ids]
    ordered_strikeout_rate = [player_batting_stats.get(p_id, {}).get('strikeout_rate', -1) for p_id in all_player_ids]

    print('Parsing allstar data...')
    allstar_data = json.load(open(args.allstar_data_fp))
    # ordered_allstar_apps = [len(allstar_data[p_id]['allstar_seasons']) if allstar_data[p_id]['pitcher'] == 'F' else 0 for p_id in all_player_ids]
    ordered_allstar_apps = [
        len(allstar_data.get(p_id, {}).get('allstar_seasons', [])) if allstar_data.get(p_id, {}).get('pitcher',
                                                                                                     'F') == 'F' else 0
        for p_id in all_player_ids]

    print('Parsing HR Crown data...')
    hr_crown_data = json.load(open(args.hr_crown_data_fp))
    ordered_hr_crowns = [len(hr_crown_data.get(p_id, {}).get('seasons', [])) for p_id in all_player_ids]

    print('Parsing Batting titles...')
    batting_title_data = json.load(open(args.batting_title_data_fp))
    ordered_batting_titles = [len(batting_title_data.get(p_id, {}).get('seasons', [])) for p_id in all_player_ids]

    print('Parsing RBI titles...')
    rbi_title_data = json.load(open(args.rbi_title_data_fp))
    ordered_rbi_titles = [len(rbi_title_data.get(p_id, {}).get('seasons', [])) for p_id in all_player_ids]

    print('Parsing Stolen Base Leaders...')
    stolen_base_data = json.load(open(args.stolen_base_leaders_data_fp))
    ordered_stolen_base_titles = [len(stolen_base_data.get(p_id, {}).get('seasons', [])) for p_id in all_player_ids]

    tsne_fp = os.path.join(bin_dir, 'tsne_{}_{}perp_{}iter_min_{}_pa.npy'.format(form_desc,
                                                                                 args.tsne_perplexity,
                                                                                 args.tsne_n_iter, args.min_pa))
    if os.path.exists(tsne_fp):
        print('Reading tSNE from file...')
        tsne_proj = np.load(tsne_fp)
    else:
        # print('Using PCA to project dim of embds to 36 from {}...'.format(all_player_embds.shape[-1]))
        # pca_comps = do_pca(all_player_embds, n=36)
        # print('\tpca_comps: {}'.format(pca_comps.shape))

        print('Performing tSNE...')
        tsne_proj = do_tsne(all_player_embds, perplexity=args.tsne_perplexity, n_iter=args.tsne_n_iter)
        print('Saving for later...')
        np.save(tsne_fp, tsne_proj)

    plt_fp = os.path.join(bin_dir, 'batter_form_{}_{}perp_{}iter_min_{}_pa.png'.format(form_desc,
                                                                                       args.tsne_perplexity,
                                                                                       args.tsne_n_iter, args.min_pa))

    xs = tsne_proj[:, 0]
    ys = tsne_proj[:, 1]

    print('Making *ALL* plots...')
    make_batter_plots(xs, ys, avg_batting_order_idx, ordered_player_salaries, ordered_player_wars, ordered_player_pos,
                      ordered_player_handedness, ordered_batting_avg, ordered_batting_avg_v_lhp,
                      ordered_batting_avg_v_rhp, ordered_n_pa, ordered_pct_rhp, ordered_avg_hit_distance,
                      ordered_avg_launch_speed, ordered_avg_launch_angle, ordered_avg_hc_x, ordered_avg_hc_y,
                      ordered_allstar_apps, ordered_single_rate, ordered_double_rate, ordered_triple_rate,
                      ordered_hr_rate, ordered_rbi_rate, ordered_walk_rate, ordered_strikeout_rate,
                      ordered_hr_crowns, ordered_batting_titles, ordered_rbi_titles, ordered_stolen_base_titles,
                      plt_fp=plt_fp)

    succinct_plt_fp = os.path.join(bin_dir, 'succinct_batter_form_{}_{}perp_{}iter_min_{}_pa.png'.format(
        form_desc, args.tsne_perplexity, args.tsne_n_iter, args.min_pa))
    print('Making *SUCCINCT* plots...')
    make_succinct_batter_plots(xs, ys, ordered_allstar_apps, ordered_player_wars, ordered_hr_rate,
                               ordered_player_handedness, ordered_player_pos, plt_fp=succinct_plt_fp)


def read_whip_era(player_ids, db_fp, bio_info_d):
    player_id_to_whip = {}
    player_id_to_era = {}
    whip_era_query = """select AVG(WHIP), AVG(ERA)
                                    from pitching_by_season
                                    where Name like ? and Season >= 2015"""

    for player_id in player_ids:
        player_name = bio_info_d[str(player_id)]['fangraphs_name']
        whip_era_args = (player_name, )
        whip_era_res = query_db(db_fp, whip_era_query, whip_era_args)

        try:
            player_whip = float(whip_era_res[0][0])
            player_era = float(whip_era_res[0][1])
        except Exception as ex:
            player_whip = -1
            player_era = -1

        player_id_to_whip[player_id] = player_whip
        player_id_to_era[player_id] = player_era

    print('len(player_id_to_whip): {}'.format(len(player_id_to_whip)))
    print('len(player_id_to_era): {}'.format(len(player_id_to_era)))

    return player_id_to_whip, player_id_to_era


def parse_pitching_opp_avg(ps_df):
    opp_avg = 0.0
    lhb_opp_avg = None
    rhb_opp_avg = None

    all_events = ps_df['events'].tolist()
    u_events, u_event_counts = np.unique(all_events, return_counts=True)
    all_event_d = dict(zip(u_events, u_event_counts))
    opp_avg = calc_batting_avg_from_event_d(all_event_d)

    events_v_lhb = ps_df[ps_df['stand'] == 'L']['events'].tolist()
    if len(events_v_lhb) > 0:
        u_events_v_lhb, u_event_counts_v_lhb = np.unique(events_v_lhb, return_counts=True)
        all_event_v_lhb_d = dict(zip(u_events_v_lhb, u_event_counts_v_lhb))
        lhb_opp_avg = calc_batting_avg_from_event_d(all_event_v_lhb_d)

    events_v_rhb = ps_df[ps_df['stand'] == 'R']['events'].tolist()
    if len(events_v_rhb) > 0:
        u_events_v_rhb, u_event_counts_v_rhb = np.unique(events_v_rhb, return_counts=True)
        all_event_v_rhb_d = dict(zip(u_events_v_rhb, u_event_counts_v_rhb))
        rhb_opp_avg = calc_batting_avg_from_event_d(all_event_v_rhb_d)

    return opp_avg, lhb_opp_avg, rhb_opp_avg


def parse_pitch_frequency(pitch_type_d):
    n_pitches = sum(pitch_type_d.values())
    fastball_pitch_types = ['FF', 'FC', 'FS', 'FT', 'FA', 'FO']
    breaking_pitch_types = ['SL', 'CU', 'KC', 'SC', 'KN', 'SI']
    changeup_pitch_types = ['CH', 'EP']
    other_pitch_types = ['PO', 'IN', 'UN']

    if n_pitches <= 0:
        pct_fastball = 0.0
        pct_breaking = 0.0
        pct_changeup = 0.0
        pct_other = 0.0
    else:
        pct_fastball = sum([pitch_type_d.get(pt, 0) for pt in fastball_pitch_types]) / n_pitches
        pct_breaking = sum([pitch_type_d.get(pt, 0) for pt in breaking_pitch_types]) / n_pitches
        pct_changeup = sum([pitch_type_d.get(pt, 0) for pt in changeup_pitch_types]) / n_pitches
        pct_other = sum([pitch_type_d.get(pt, 0) for pt in other_pitch_types]) / n_pitches

    return pct_fastball, pct_breaking, pct_changeup, pct_other


def calc_player_pitching_stats(player_ids, db_fp):
    pitching_stats = {}
    ps_query = """select pitch_type, release_speed, release_spin_rate, release_extension,
                        hit_distance_sc, launch_speed, launch_angle, game_pk, at_bat_number, stand, events
                      from statcast
                      where game_year >= 2015 and game_year <= 2019 and pitcher = ?"""

    for player_id in player_ids:
        ps_args = (player_id,)
        ps_res = query_db(db_fp, ps_query, ps_args)
        ps_df = pd.DataFrame(ps_res, columns=['pitch_type', 'release_speed', 'release_spin_rate', 'release_extension',
                                              'hit_distance_sc', 'launch_speed', 'launch_angle', 'game_pk',
                                              'at_bat_number', 'stand', 'events'])
        player_game_pks = ps_df['game_pk'].tolist()
        player_ab_nos = ps_df['at_bat_number'].tolist()
        player_pa_ids = ['{}-{}'.format(g_pk, ab_no) for g_pk, ab_no in zip(player_game_pks, player_ab_nos)]
        u_player_pa_ids = list(np.unique(player_pa_ids))

        ps_df = ps_df[ps_df['events'].notnull()]
        opp_avg, lhb_opp_avg, rhb_opp_avg = parse_pitching_opp_avg(ps_df)

        pitch_types = [pt for pt in ps_df['pitch_type'].tolist() if pt is not None]
        pitch_types, pitch_type_counts = np.unique(pitch_types, return_counts=True)
        # n_pitch_types = len(list(pitch_types))
        n_pitch_types = len([pt for pt, ptc in zip(pitch_types, pitch_type_counts) if ptc > 5])
        pitch_type_d = dict(zip(pitch_types, pitch_type_counts))
        fastball_pct, breaking_pct, changeup_pct, other_pct = parse_pitch_frequency(pitch_type_d)

        release_speeds = ps_df['release_speed'].tolist()
        release_spin_rates = ps_df['release_spin_rate'].tolist()
        release_extensions = ps_df['release_extension'].tolist()
        hit_distances = ps_df['hit_distance_sc'].tolist()
        launch_speeds = ps_df['launch_speed'].tolist()
        launch_angles = ps_df['launch_angle'].tolist()

        release_speeds = [x for x in release_speeds if x is not None and str(x) != 'nan']
        release_spin_rates = [x for x in release_spin_rates if x is not None and str(x) != 'nan']
        release_extensions = [x for x in release_extensions if x is not None and str(x) != 'nan']
        hit_distances = [x for x in hit_distances if x is not None and str(x) != 'nan']
        launch_speeds = [x for x in launch_speeds if x is not None and str(x) != 'nan']
        launch_angles = [x for x in launch_angles if x is not None and str(x) != 'nan']

        avg_release_speed = sum(release_speeds) / len(release_speeds) if len(release_speeds) > 0 else 0.0
        avg_spin_rate = sum(release_spin_rates) / len(release_spin_rates) if len(release_spin_rates) > 0 else 0.0
        avg_extension = sum(release_extensions) / len(release_extensions) if len(release_extensions) > 0 else 0.0
        avg_hit_distance = sum(hit_distances) / len(hit_distances) if len(hit_distances) > 0 else 0.0
        avg_launch_speed = sum(launch_speeds) / len(launch_speeds) if len(launch_speeds) > 0 else 0.0
        avg_launch_angle = sum(launch_angles) / len(launch_angles) if len(launch_angles) > 0 else 0.0

        pitching_stats[player_id] = {
            'n_batters_faced': len(u_player_pa_ids), 'arsenal_size': n_pitch_types,
            'avg_release_speed': avg_release_speed, 'avg_spin_rate': avg_spin_rate,
            'avg_extension': avg_extension, 'avg_hit_distance': avg_hit_distance,
            'avg_launch_speed': avg_launch_speed, 'avg_launch_angle': avg_launch_angle,
            'opp_avg': opp_avg, 'lhb_opp_avg': lhb_opp_avg,
            'rhb_opp_avg': rhb_opp_avg, 'fastball_pct': fastball_pct,
            'breaking_pct': breaking_pct, 'changeup_pct': changeup_pct, 'other_pct': other_pct,
        }
    return pitching_stats


def make_pitcher_plots(xs, ys, ordered_player_salaries, ordered_player_wars, ordered_player_pos,
                       ordered_player_handedness, allstar_apps, ordered_n_batters_faced, ordered_arsenal_size,
                       ordered_avg_release_speed, ordered_avg_spin_rate, ordered_avg_extension,
                       ordered_avg_hit_distance, ordered_avg_launch_speed, ordered_avg_launch_angle,
                       ordered_opp_avg, ordered_lhb_opp_avg, ordered_rhb_opp_avg, ordered_fastball_pct,
                       ordered_breaking_pct, ordered_changeup_pct, ordered_other_pct, ordered_whip, ordered_era,
                       plt_fp):
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow', 'brown', 'black', 'skyblue',
              'chartreuse',
              'sienna', 'lavender', 'olive']
    n_cols = 5
    n_rows = 5

    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
    # cmap = plt.cm.cool
    # cmap = plt.cm.Blues
    cmap = plt.cm.hot

    print('Plotting salary...')
    plt.subplot(n_rows, n_cols, 1)
    salary_xs = [xs[i] for i in range(len(ordered_player_salaries)) if ordered_player_salaries[i] is not None]
    salary_ys = [ys[i] for i in range(len(ordered_player_salaries)) if ordered_player_salaries[i] is not None]
    ordered_player_salaries = [ops for ops in ordered_player_salaries if ops is not None]
    plt.scatter(salary_xs, salary_ys, c=ordered_player_salaries, cmap=cmap, alpha=0.5)
    plt.title('Salary')
    plt.colorbar()

    print('Plotting WAR...')
    plt.subplot(n_rows, n_cols, 2)
    war_xs = [xs[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and ordered_player_pos[i] in ['P', 'SP', 'RP']]
    war_ys = [ys[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and 'P' in ordered_player_pos[i]]
    ordered_player_wars = [opw for i, opw in enumerate(ordered_player_wars) if
                           opw is not None and 'P' in ordered_player_pos[i]]
    plt.scatter(war_xs, war_ys, c=ordered_player_wars, cmap=cmap, alpha=0.5)
    plt.title('War')
    plt.colorbar()

    print('Plotting position 1...')
    plt.subplot(n_rows, n_cols, 3)
    unique_player_positions = list(np.unique(ordered_player_pos))
    unique_player_positions = [upp for upp in unique_player_positions if upp != 'UNK']
    print('unique_player_positions: {}'.format(unique_player_positions))
    for pos_idx, pos_label in enumerate(unique_player_positions):
        pos_xs = [xs[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
        pos_ys = [ys[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
        pos_color = colors[pos_idx]
        plt.scatter(pos_xs, pos_ys, c=pos_color, label=pos_label, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Position (Raw)')

    # ordered_player_pos = [quick_map(opp) for opp in ordered_player_pos]
    # print('Plotting position 2...')
    # plt.subplot(n_rows, n_cols, 4)
    # unique_player_positions = list(np.unique(ordered_player_pos))
    # unique_player_positions = [upp for upp in unique_player_positions if upp != 'UNK']
    # print('unique_player_positions: {}'.format(unique_player_positions))
    # for pos_idx, pos_label in enumerate(unique_player_positions):
    #     pos_xs = [xs[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
    #     pos_ys = [ys[i] for i in range(len(ordered_player_pos)) if ordered_player_pos[i] == pos_label]
    #     pos_color = colors[pos_idx]
    #     plt.scatter(pos_xs, pos_ys, c=pos_color, label=pos_label, alpha=0.5)
    # plt.legend(bbox_to_anchor=(1, 1))
    # plt.title('Position (Rough)')

    print('Plotting handedness...')
    plt.subplot(n_rows, n_cols, 4)
    for hand_idx, hand_label in enumerate(['R', 'L']):
        hand_xs = [xs[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_ys = [ys[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_color = colors[hand_idx]
        plt.scatter(hand_xs, hand_ys, c=hand_color, label=hand_label, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Handedness')

    print('Plotting all star apps...')
    plt.subplot(n_rows, n_cols, 5)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        as_ys = [ys[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]
        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('All-star Apps')

    print('Plotting arsenal size...')
    plt.subplot(n_rows, n_cols, 6)
    plt.scatter(xs, ys, c=ordered_arsenal_size, cmap=cmap, alpha=0.5)
    plt.title('Arsenal Size')
    plt.colorbar()

    print('Plotting pct fastball..')
    plt.subplot(n_rows, n_cols, 7)
    plt.scatter(xs, ys, c=ordered_fastball_pct, cmap=cmap, alpha=0.5)
    plt.title('Fastball %')
    plt.colorbar()

    print('Plotting pct breaking ball..')
    plt.subplot(n_rows, n_cols, 8)
    plt.scatter(xs, ys, c=ordered_breaking_pct, cmap=cmap, alpha=0.5)
    plt.title('Breaking Ball %')
    plt.colorbar()

    print('Plotting pct changeup..')
    plt.subplot(n_rows, n_cols, 9)
    plt.scatter(xs, ys, c=ordered_changeup_pct, cmap=cmap, alpha=0.5)
    plt.title('Changeup %')
    plt.colorbar()

    print('Plotting pct other..')
    plt.subplot(n_rows, n_cols, 10)
    plt.scatter(xs, ys, c=ordered_other_pct, cmap=cmap, alpha=0.5)
    plt.title('Other %')
    plt.colorbar()

    print('Plotting release speed...')
    plt.subplot(n_rows, n_cols, 11)
    plt.scatter(xs, ys, c=ordered_avg_release_speed, cmap=cmap, alpha=0.5)
    plt.title('Release Speed')
    plt.colorbar()

    print('Plotting spin rate...')
    plt.subplot(n_rows, n_cols, 12)
    plt.scatter(xs, ys, c=ordered_avg_spin_rate, cmap=cmap, alpha=0.5)
    plt.title('Spin Rate')
    plt.colorbar()

    print('Plotting extension...')
    plt.subplot(n_rows, n_cols, 13)
    plt.scatter(xs, ys, c=ordered_avg_extension, cmap=cmap, alpha=0.5)
    plt.title('Extension')
    plt.colorbar()

    print('Plotting hit distance...')
    plt.subplot(n_rows, n_cols, 14)
    plt.scatter(xs, ys, c=ordered_avg_hit_distance, cmap=cmap, alpha=0.5)
    plt.title('Hit Distance')
    plt.colorbar()

    print('Plotting launch speed...')
    plt.subplot(n_rows, n_cols, 15)
    plt.scatter(xs, ys, c=ordered_avg_launch_speed, cmap=cmap, alpha=0.5)
    plt.title('Launch Speed')
    plt.colorbar()

    print('Plotting launch angle...')
    plt.subplot(n_rows, n_cols, 16)
    plt.scatter(xs, ys, c=ordered_avg_launch_angle, cmap=cmap, alpha=0.5)
    plt.title('Launch Angle')
    plt.colorbar()

    print('Plotting # batters faced...')
    plt.subplot(n_rows, n_cols, 17)
    plt.scatter(xs, ys, c=ordered_n_batters_faced, cmap=cmap, alpha=0.5)
    plt.title('Batters Faced')
    plt.colorbar()

    print('Plotting opp avg...')
    plt.subplot(n_rows, n_cols, 18)
    plt.scatter(xs, ys, c=ordered_opp_avg, cmap=cmap, alpha=0.5)
    plt.title('Opponent Avg')
    plt.colorbar()

    print('Plotting lhb opp avg...')
    plt.subplot(n_rows, n_cols, 19)
    plt.scatter(xs, ys, c=ordered_lhb_opp_avg, cmap=cmap, alpha=0.5)
    plt.title('Opponent LHB Avg')
    plt.colorbar()

    print('Plotting rhb opp avg...')
    plt.subplot(n_rows, n_cols, 20)
    plt.scatter(xs, ys, c=ordered_rhb_opp_avg, cmap=cmap, alpha=0.5)
    plt.title('Opponent RHB Avg')
    plt.colorbar()

    print('Plotting WHIP...')
    whip_xs = [xs[i] for i in range(len(ordered_whip)) if ordered_whip[i] != -1]
    whip_ys = [ys[i] for i in range(len(ordered_whip)) if ordered_whip[i] != -1]
    ordered_whip = [ow for ow in ordered_whip if ow != -1]
    plt.subplot(n_rows, n_cols, 21)
    plt.scatter(whip_xs, whip_ys, c=ordered_whip, cmap=cmap, alpha=0.5)
    plt.title('WHIP')
    plt.colorbar()

    print('Plotting ERA...')
    era_xs = [xs[i] for i in range(len(ordered_era)) if ordered_era[i] != -1 and ordered_era[i] < 20]
    era_ys = [ys[i] for i in range(len(ordered_era)) if ordered_era[i] != -1 and ordered_era[i] < 20]
    ordered_era = [oe for oe in ordered_era if oe != -1 and oe < 20]
    plt.subplot(n_rows, n_cols, 22)
    plt.scatter(era_xs, era_ys, c=ordered_era, cmap=cmap, alpha=0.5)
    plt.title('ERA (< 20)')
    plt.colorbar()

    plt.suptitle('Pitcher Form tSNE')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.2)
    plt.savefig(plt_fp, bbox_inches='tight')
    plt.clf()


def make_succinct_pitcher_plots(xs, ys, allstar_apps, ordered_player_wars, ordered_breaking_pct,
                                ordered_player_handedness, ordered_player_pos, plt_fp):
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'chartreuse', 'pink', 'yellow', 'black',
              'skyblue', 'red', 'lavender', 'olive']
    cmap = plt.cm.YlGnBu
    n_cols = 4
    n_rows = 1

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))

    print('Plotting all star apps...')
    plt.subplot(n_rows, n_cols, 1)
    for as_idx, as_label in enumerate([0, 1, 2, 3, 4, 5]):
        as_xs = [xs[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        as_ys = [ys[i] for i in range(len(allstar_apps)) if allstar_apps[i] == as_label]
        if as_idx == 0:
            as_color = 'lightgrey'
        elif as_idx == 5:
            as_color = 'red'
        else:
            as_color = colors[as_idx]
        if len(as_xs) > 0:
            plt.scatter(as_xs, as_ys, c=as_color, label=str(as_label), alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('All-star Apps')

    print('Plotting WAR...')
    plt.subplot(n_rows, n_cols, 2)
    war_xs = [xs[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and ordered_player_pos[i] in ['P', 'SP', 'RP']]
    war_ys = [ys[i] for i in range(len(ordered_player_wars)) if
              ordered_player_wars[i] is not None and 'P' in ordered_player_pos[i]]
    ordered_player_wars = [opw for i, opw in enumerate(ordered_player_wars) if
                           opw is not None and 'P' in ordered_player_pos[i]]
    plt.scatter(war_xs, war_ys, c=ordered_player_wars, cmap=cmap, alpha=0.5)
    plt.title('War')
    plt.colorbar()

    print('Plotting pct breaking ball..')
    plt.subplot(n_rows, n_cols, 3)
    plt.scatter(xs, ys, c=ordered_breaking_pct, cmap=cmap, alpha=0.5)
    plt.title('Breaking Ball %')
    plt.colorbar()

    print('Plotting handedness...')
    plt.subplot(n_rows, n_cols, 4)
    for hand_idx, hand_label in enumerate(['R', 'L']):
        hand_xs = [xs[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_ys = [ys[i] for i in range(len(ordered_player_handedness)) if ordered_player_handedness[i] == hand_label]
        hand_color = colors[hand_idx]
        plt.scatter(hand_xs, hand_ys, c=hand_color, label=hand_label, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Handedness')

    plt.suptitle('Pitcher Form tSNE', y=1.0)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.2)
    plt.savefig(plt_fp, bbox_inches='tight')
    plt.clf()


def inspect_pitcher_form_embeddings(args, form_desc, all_form_embds, player_bio_info):
    player_salary_fp = os.path.join(bin_dir, 'player_salary_{}.json'.format(form_desc))
    player_war_fp = os.path.join(bin_dir, 'player_war_{}.json'.format(form_desc))
    player_pos_fp = os.path.join(bin_dir, 'player_pos_{}.json'.format(form_desc))
    player_handedness_fp = os.path.join(bin_dir, 'player_handedness_{}.json'.format(form_desc))
    player_pitching_stats_fp = os.path.join(bin_dir, 'player_pitching_stats_{}.json'.format(form_desc))
    player_whip_fp = os.path.join(bin_dir, 'player_whip_{}.json'.format(form_desc))
    player_era_fp = os.path.join(bin_dir, 'player_era_{}.json'.format(form_desc))
    ordered_batter_ids_fp = os.path.join(bin_dir, 'ordered_batter_ids_{}.json'.format(form_desc))

    if os.path.exists(player_salary_fp) and os.path.exists(player_war_fp) and os.path.exists(player_pos_fp) \
            and os.path.exists(player_handedness_fp):
        print('Reading meta data...')
        player_salary = json.load(open(player_salary_fp))
        player_war = json.load(open(player_war_fp))
        player_pos = json.load(open(player_pos_fp))
        player_handedness = json.load(open(player_handedness_fp))
    else:
        print('Computing meta data...')
        all_player_ids = list(all_form_embds.keys())
        player_salary, player_war, player_pos, player_handedness = read_salary_war_pos_hand(all_player_ids,
                                                                                            args.db_fp, player_bio_info,
                                                                                            player_type='pitcher')
        with open(player_salary_fp, 'w+') as f:
            f.write(json.dumps(player_salary, indent=1))

        with open(player_war_fp, 'w+') as f:
            f.write(json.dumps(player_war, indent=1))

        with open(player_pos_fp, 'w+') as f:
            f.write(json.dumps(player_pos, indent=1))

        with open(player_handedness_fp, 'w+') as f:
            f.write(json.dumps(player_handedness, indent=1))

    if os.path.exists(player_whip_fp) and os.path.exists(player_era_fp):
        print('Reading WHIP and ERA data...')
        player_whip = json.load(open(player_whip_fp))
        player_era = json.load(open(player_era_fp))
    else:
        print('Computing WHIP and ERA data...')
        all_player_ids = list(all_form_embds.keys())
        player_whip, player_era = read_whip_era(all_player_ids, args.db_fp, player_bio_info)

    all_player_ids = list(all_form_embds.keys())
    if os.path.exists(player_pitching_stats_fp):
        print('Reading pitching stats from file...')
        player_pitching_stats = json.load(open(player_pitching_stats_fp))
    else:
        print('Calculating player pitching stats...')
        player_pitching_stats = calc_player_pitching_stats(all_player_ids, args.db_fp)

        with open(player_pitching_stats_fp, 'w+') as f:
            f.write(json.dumps(player_pitching_stats, indent=1))

    with open(ordered_batter_ids_fp, 'w+') as f:
        f.write(json.dumps(all_player_ids, indent=1))
    all_player_embds = np.vstack([all_form_embds[p_id] for p_id in all_player_ids])
    ordered_player_salaries = [player_salary.get(p_id, None) for p_id in all_player_ids]
    ordered_player_wars = [player_war.get(p_id, None) for p_id in all_player_ids]
    ordered_player_pos = [player_pos[p_id] for p_id in all_player_ids]
    ordered_player_pos = [p_pos if str(p_pos) != 'nan' else 'UNK' for p_pos in ordered_player_pos]
    ordered_player_handedness = [player_handedness.get(p_id, None) for p_id in all_player_ids]
    ordered_n_batters_faced = [player_pitching_stats.get(p_id, {}).get('n_batters_faced', -1) for p_id in
                               all_player_ids]
    ordered_arsenal_size = [player_pitching_stats.get(p_id, {}).get('arsenal_size', -1) for p_id in all_player_ids]
    ordered_avg_release_speed = [player_pitching_stats.get(p_id, {}).get('avg_release_speed', -1) for p_id in
                                 all_player_ids]
    ordered_avg_spin_rate = [player_pitching_stats.get(p_id, {}).get('avg_spin_rate', -1) for p_id in all_player_ids]
    ordered_avg_extension = [player_pitching_stats.get(p_id, {}).get('avg_extension', -1) for p_id in all_player_ids]
    ordered_avg_hit_distance = [player_pitching_stats.get(p_id, {}).get('avg_hit_distance', -1) for p_id in
                                all_player_ids]
    ordered_avg_launch_speed = [player_pitching_stats.get(p_id, {}).get('avg_launch_speed', -1) for p_id in
                                all_player_ids]
    ordered_avg_launch_angle = [player_pitching_stats.get(p_id, {}).get('avg_launch_angle', -1) for p_id in
                                all_player_ids]
    ordered_opp_avg = [player_pitching_stats.get(p_id, {}).get('opp_avg', -1) for p_id in all_player_ids]
    ordered_lhb_opp_avg = [player_pitching_stats.get(p_id, {}).get('lhb_opp_avg', -1) for p_id in all_player_ids]
    ordered_rhb_opp_avg = [player_pitching_stats.get(p_id, {}).get('rhb_opp_avg', -1) for p_id in all_player_ids]
    ordered_fastball_pct = [player_pitching_stats.get(p_id, {}).get('fastball_pct', 0) for p_id in all_player_ids]
    ordered_breaking_pct = [player_pitching_stats.get(p_id, {}).get('breaking_pct', 0) for p_id in all_player_ids]
    ordered_changeup_pct = [player_pitching_stats.get(p_id, {}).get('changeup_pct', 0) for p_id in all_player_ids]
    ordered_other_pct = [player_pitching_stats.get(p_id, {}).get('other_pct', 0) for p_id in all_player_ids]
    ordered_whip = [player_whip.get(p_id, -1) for p_id in all_player_ids]
    ordered_era = [player_era.get(p_id, -1) for p_id in all_player_ids]

    # print('ordered_player_wars: {}'.format(ordered_player_wars))
    # print('ordered_player_pos: {}'.format(ordered_player_pos))

    print('Parsing allstar data...')
    allstar_data = json.load(open(args.allstar_data_fp))
    ordered_allstar_apps = [
        len(allstar_data.get(p_id, {}).get('allstar_seasons', [])) if allstar_data.get(p_id, {}).get('pitcher',
                                                                                                     'F') == 'T' else 0
        for p_id in all_player_ids]
    tsne_fp = os.path.join(bin_dir, 'tsne_{}_{}perp_{}iter.npy'.format(form_desc, args.tsne_perplexity,
                                                                       args.tsne_n_iter))

    if os.path.exists(tsne_fp):
        print('Reading tSNE from file...')
        tsne_proj = np.load(tsne_fp)
    else:
        print('Using PCA to project dim of embds to 36 from {}...'.format(all_player_embds.shape[-1]))
        pca_comps = do_pca(all_player_embds, n=36)
        print('\tpca_comps: {}'.format(pca_comps.shape))

        print('Performing tSNE...')
        tsne_proj = do_tsne(pca_comps, perplexity=args.tsne_perplexity, n_iter=args.tsne_n_iter)
        print('Saving for later...')
        np.save(tsne_fp, tsne_proj)

    plt_fp = os.path.join(bin_dir, 'pitcher_form_{}_{}perp_{}iter'.format(form_desc, args.tsne_perplexity,
                                                                          args.tsne_n_iter))
    print('*** plt_fp: {} ***'.format(plt_fp))
    xs = tsne_proj[:, 0]
    ys = tsne_proj[:, 1]
    print('Making plots...')
    make_pitcher_plots(xs, ys, ordered_player_salaries, ordered_player_wars, ordered_player_pos,
                       ordered_player_handedness, ordered_allstar_apps, ordered_n_batters_faced, ordered_arsenal_size,
                       ordered_avg_release_speed, ordered_avg_spin_rate, ordered_avg_extension,
                       ordered_avg_hit_distance, ordered_avg_launch_speed, ordered_avg_launch_angle,
                       ordered_opp_avg, ordered_lhb_opp_avg, ordered_rhb_opp_avg, ordered_fastball_pct,
                       ordered_breaking_pct, ordered_changeup_pct, ordered_other_pct,
                       ordered_whip, ordered_era,
                       plt_fp=plt_fp)

    succinct_plt_fp = os.path.join(bin_dir, 'succinct_pitcher_form_{}_{}perp_{}iter'.format(
        form_desc, args.tsne_perplexity, args.tsne_n_iter))
    print('Making *SUCCINCT* plots...')
    make_succinct_pitcher_plots(xs, ys, ordered_allstar_apps, ordered_player_wars, ordered_breaking_pct,
                                ordered_player_handedness, ordered_player_pos, plt_fp=succinct_plt_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--form_rep_dir',
                        default='/home/czh/sata1/SportsAnalytics/batter_form_v46')
    parser.add_argument('--whole_game_records_dir',
                        default='/home/czh/sata1/SportsAnalytics/whole_game_records/by_season')
    parser.add_argument('--db_fp', default='../database/mlb.db')
    parser.add_argument('--statcast_id_to_bio_info_fp', default='../config/statcast_id_to_bio_info.json')
    parser.add_argument('--allstar_data_fp', default='../config/allstar_data.json')
    parser.add_argument('--hr_crown_data_fp', default='../config/hr_crown_data.json')
    parser.add_argument('--batting_title_data_fp', default='../config/batting_title_data.json')
    parser.add_argument('--rbi_title_data_fp', default='../config/rbi_title_data.json')
    parser.add_argument('--stolen_base_leaders_data_fp', default='../config/stolen_base_leaders.json')

    parser.add_argument('--n_workers', default=12, type=int)
    parser.add_argument('--tsne_perplexity', default=75, type=int)
    parser.add_argument('--tsne_n_iter', default=1000, type=int)
    parser.add_argument('--min_pa', default=15, type=int)

    parser.add_argument('--stats_mode', default=False, type=str2bool)
    parser.add_argument('--data_scopes_to_use',
                        default=['career', 'season', 'last15'], type=str, nargs='+',
                        help='What data scopes to use when constructing stat-based embeddings')

    args = parser.parse_args()

    bin_dir = os.path.join(args.form_rep_dir, 'bin')
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    form_desc = os.path.split(args.form_rep_dir)[-1]
    all_form_embds_fp = os.path.join(bin_dir, 'all_form_embds_{}.json'.format(form_desc))
    batting_order_ids_fp = os.path.join(bin_dir, 'batting_order_ids_{}.json'.format(form_desc))
    player_bio_info = json.load(open(args.statcast_id_to_bio_info_fp))

    if os.path.exists(all_form_embds_fp) and os.path.exists(batting_order_ids_fp):
        print('Reading agg arrays from file...')
        all_form_embds = json.load(open(all_form_embds_fp))
        batting_order_ids = json.load(open(batting_order_ids_fp))
    else:
        print('Computing agg arrays...')
        all_form_embds, batting_order_ids = read_embds(args.form_rep_dir, args.whole_game_records_dir, args.n_workers)
        print('Saving arrays to file for later...')
        with open(all_form_embds_fp, 'w+') as f:
            f.write(json.dumps(all_form_embds, indent=1))

        with open(batting_order_ids_fp, 'w+') as f:
            f.write(json.dumps(batting_order_ids, indent=1))

    if args.stats_mode:
        print('*** REPLACING FORM EMBDS WITH STAT EMBDS ***')
        record_norm_values_fp = os.path.join(os.path.split(args.whole_game_records_dir)[0],
                                             'game_event_splits', 'stats', 'max_values.json')
        record_norm_values = json.load(open(record_norm_values_fp))
        stats_fp = os.path.join(bin_dir, 'batter_stats.json' if 'batter' in form_desc else 'pitcher_stats.json')
        form_desc = 'batter_STATS' if 'batter' in form_desc else 'pitcher_STATS'
        if os.path.exists(stats_fp):
            print('Reading pre-computed stats from file...')
            all_form_embds = json.load(open(stats_fp))
        else:
            print('Calculating stats...')
            all_form_embds = read_starter_stats(args.whole_game_records_dir, record_norm_values,
                                                player_type='batter' if 'batter' in form_desc else 'pitcher',
                                                data_scopes_to_use=args.data_scopes_to_use)
            with open(stats_fp, 'w+') as f:
                f.write(json.dumps(all_form_embds, indent=1))

    if 'batter' in form_desc:
        print('Inspecting batter form embeddings...')
        inspect_batter_form_embeddings(args, form_desc, all_form_embds, batting_order_ids, player_bio_info)
    else:
        print('Inspecting pitcher form embeddings...')
        inspect_pitcher_form_embeddings(args, form_desc, all_form_embds, player_bio_info)
