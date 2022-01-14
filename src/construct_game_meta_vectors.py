__author__ = 'Connor Heaton'

import os
import json
import sqlite3
import argparse


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


def construct_team_meta_data(db_fp):
    team_meta_stats = {}

    all_game_pk_query = """select distinct(game_pk) from statcast where game_year >= 2015 and game_year <= 2019"""
    all_game_pks = [x[0] for x in query_db(db_fp, all_game_pk_query)]
    # print('len(all_game_pks): {}'.format(len(all_game_pks)))

    for game_idx, game_pk in enumerate(all_game_pks):
        if game_idx % 1000 == 0:
            print('Analyzing game {}/{}...'.format(game_idx, len(all_game_pks)))

        meta_attr_names = ['game_pk', 'home_team', 'away_team', 'home_score', 'away_score', 'inning', 'inning_topbot',
                           'game_year']
        game_meta_query = """select game_pk, home_team, away_team, home_score, away_score, inning, inning_topbot, 
                                    game_year 
                                 from statcast
                                 where game_pk = ?
                                 order by at_bat_number desc limit 1"""
        game_meta_args = (game_pk,)
        game_meta_res = query_db(db_fp, game_meta_query, game_meta_args)[0]
        game_meta_d = dict(zip(meta_attr_names, game_meta_res))

        if game_meta_d['away_score'] > game_meta_d['home_score']:
            home_win_val = 0
            away_win_val = 1

            home_rs = game_meta_d['home_score']
            away_rs = game_meta_d['away_score']
        else:
            home_win_val = 1
            away_win_val = 0
            if game_meta_d['away_score'] == game_meta_d['home_score']:
                game_meta_d['home_score'] += 1

            home_rs = game_meta_d['home_score']
            away_rs = game_meta_d['away_score']

        # Update meta for home team
        home_team_meta = team_meta_stats.get(game_meta_d['home_team'], {})
        home_team_season_meta = home_team_meta.get(game_meta_d['game_year'],
                                                   {'game_pk': [], 'win_status': [], 'runs_scored': [],
                                                    'runs_allowed': []})
        home_team_season_meta['game_pk'].append(game_pk)
        home_team_season_meta['win_status'].append(home_win_val)
        home_team_season_meta['runs_scored'].append(home_rs)
        home_team_season_meta['runs_allowed'].append(away_rs)
        home_team_meta[game_meta_d['game_year']] = home_team_season_meta
        team_meta_stats[game_meta_d['home_team']] = home_team_meta

        # Update meta for away team
        away_team_meta = team_meta_stats.get(game_meta_d['away_team'], {})
        away_team_season_meta = away_team_meta.get(game_meta_d['game_year'],
                                                   {'game_pk': [], 'win_status': [], 'runs_scored': [],
                                                    'runs_allowed': []})
        away_team_season_meta['game_pk'].append(game_pk)
        away_team_season_meta['win_status'].append(home_win_val)
        away_team_season_meta['runs_scored'].append(home_rs)
        away_team_season_meta['runs_allowed'].append(away_rs)
        away_team_meta[game_meta_d['game_year']] = away_team_season_meta
        team_meta_stats[game_meta_d['away_team']] = away_team_meta

    return team_meta_stats


def log5(pa, pb):
    if (pa + pb - 2 * pa * pb) == 0.0:
        p = 0.5
    else:
        p = (pa - pa * pb) / (pa + pb - 2 * pa * pb)

    return p


def calc_meta_stats(home_team, away_team, team_meta_stats, season, game_pk):
    home_season_meta = team_meta_stats[home_team][int(season)]
    away_season_meta = team_meta_stats[away_team][int(season)]

    home_game_idx = home_season_meta['game_pk'].index(game_pk)
    away_game_idx = away_season_meta['game_pk'].index(game_pk)

    home_runs_scored = sum(home_season_meta['runs_scored'][:home_game_idx])
    home_runs_allowed = sum(home_season_meta['runs_allowed'][:home_game_idx])
    if len(home_season_meta['win_status'][:home_game_idx]) == 0:
        home_win_pct = 0.0
    else:
        home_win_pct = sum(home_season_meta['win_status'][:home_game_idx]) / len(
            home_season_meta['win_status'][:home_game_idx])
    if home_game_idx == 0:
        home_won_last_game = 0
    else:
        home_won_last_game = 1 if home_season_meta['win_status'][home_game_idx - 1] == 1 else 0

    away_runs_scored = sum(away_season_meta['runs_scored'][:away_game_idx])
    away_runs_allowed = sum(away_season_meta['runs_allowed'][:away_game_idx])
    if len(away_season_meta['win_status'][:away_game_idx]) == 0:
        away_win_pct = 0.0
    else:
        away_win_pct = sum(away_season_meta['win_status'][:away_game_idx]) / len(
            away_season_meta['win_status'][:away_game_idx])

    if away_game_idx == 0:
        away_won_last_game = 0
    else:
        away_won_last_game = 1 if away_season_meta['win_status'][away_game_idx - 1] == 1 else 0

    if (home_runs_scored ** 2 + home_runs_allowed ** 2) == 0:
        home_pe = 0.5
    else:
        home_pe = home_runs_scored ** 2 / (home_runs_scored ** 2 + home_runs_allowed ** 2)

    if (away_runs_scored ** 2 + away_runs_allowed ** 2) == 0:
        away_pe = 0.5
    else:
        away_pe = away_runs_scored ** 2 / (away_runs_scored ** 2 + away_runs_allowed ** 2)
    pe_diff = home_pe - away_pe

    home_log5 = log5(home_win_pct, away_win_pct)
    away_log5 = log5(away_win_pct, home_win_pct)
    log5_diff = home_log5 - away_log5

    wp_diff = home_win_pct - away_win_pct
    rc_diff = home_runs_scored - away_runs_scored

    return home_won_last_game, away_won_last_game, pe_diff, log5_diff, wp_diff, rc_diff


def create_meta_vectors(team_meta_data, whole_game_record_dir):
    meta_vectors = {}
    seasons = ['2015', '2016', '2017', '2018', '2019']
    # whole_game_record_dir = os.path.join(whole_game_record_dir, 'by_season')
    max_rc_diff = 1

    for season in seasons:
        print('Processing {} season...'.format(season))
        season_dir = os.path.join(whole_game_record_dir, season)

        for game_summary_file in os.listdir(season_dir):
            game_pk = int(game_summary_file[:-5])
            game_j = json.load(open(os.path.join(season_dir, game_summary_file)))
            home_team = game_j['home_team']
            away_team = game_j['away_team']

            home_won_last_game, away_won_last_game, pe_diff, log5_diff, wp_diff, rc_diff = calc_meta_stats(home_team,
                                                                                                           away_team,
                                                                                                           team_meta_data,
                                                                                                           season,
                                                                                                           game_pk)
            # make note of max rc diff, scale vectors later
            abs_rc_diff = abs(rc_diff)
            if abs_rc_diff > max_rc_diff:
                max_rc_diff = abs_rc_diff

            meta_vectors[game_pk] = [home_won_last_game, away_won_last_game, pe_diff, log5_diff, wp_diff, rc_diff]

    for game_pk in meta_vectors.keys():
        game_vector = meta_vectors[game_pk]
        game_vector[-1] = game_vector[-1] / max_rc_diff
        meta_vectors[game_pk] = game_vector

    return meta_vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--db_fp', default='../database/mlb.db')
    parser.add_argument('--out', default='/home/czh/sata1/learning_player_form/game_meta_vectors_v1')
    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records/by_season')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    out_fp = os.path.join(args.out, 'game_meta_vectors.json')

    team_meta_data = construct_team_meta_data(args.db_fp)
    game_meta_vectors = create_meta_vectors(team_meta_data, args.whole_game_record_dir)
    with open(out_fp, 'w+') as f:
        f.write(json.dumps(game_meta_vectors, indent=2))
