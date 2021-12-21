__author__ = 'Connor Heaton'

import os
import sqlite3
import argparse

import numpy as np
from sklearn.model_selection import train_test_split


def query_db(db_fp, query, args=None):
    conn = sqlite3.connect(db_fp, check_same_thread=False)
    c = conn.cursor()
    if args is None:
        c.execute(query)
    else:
        c.execute(query, args)
    rows = c.fetchall()

    return rows


def get_player_ids(db_fp, player_type, year_start):
    if player_type == 'batter':
        query = """select distinct batter 
                   from statcast 
                   where batter is not null
                   and game_year >= ?;"""
    else:
        query = """select distinct pitcher 
                   from statcast 
                   where pitcher is not null
                   and game_year >= ?;"""

    query_args = (year_start, )
    query_res = query_db(db_fp, query, query_args)
    player_ids = [qr[0] for qr in query_res]

    return player_ids


def construct_player_career_records(player_ids, player_type, year_start, db_fp, outdir):
    out_fp_tmplt = os.path.join(outdir, '{}.txt')
    if player_type == 'batter':
        query = """select game_year, game_pk, at_bat_number
                   from statcast 
                   where pitch_number=1 and batter=? and game_year>=?
                   order by game_year asc, game_pk asc, at_bat_number asc;"""
    else:
        query = """select game_year, game_pk, at_bat_number
                   from statcast 
                   where pitch_number=1 and pitcher=? and game_year>=?
                   order by game_year asc, game_pk asc, at_bat_number asc;"""

    total_n_abs = 0
    for player_id in player_ids:
        out_fp = out_fp_tmplt.format(player_id)
        query_args = (player_id, year_start,)
        query_res = query_db(db_fp, query, query_args)
        player_records = ['{}/{}-{}.json'.format(qr[0], qr[1], qr[2]) for qr in query_res]
        total_n_abs += len(player_records)
        player_write_str = '\n'.join(player_records)

        with open(out_fp, 'w+') as f:
            f.write(player_write_str)

    return total_n_abs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='/home/czh/sata1/learning_player_form/player_career_data')
    parser.add_argument('--db_fp', default='../database/mlb.db')
    parser.add_argument('--player_type', default='pitcher')
    parser.add_argument('--year_start', default=2015, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    ptype_outdir = os.path.join(args.outdir, str(args.player_type))
    if not os.path.exists(ptype_outdir):
        os.makedirs(ptype_outdir)

    print('Getting player IDs for player type = {} and start season = {}...'.format(args.player_type, args.year_start))
    player_ids = get_player_ids(args.db_fp, args.player_type, args.year_start)
    print('\tFound a total of {} player IDs...'.format(len(player_ids)))

    print('Building player career data...')
    n_abs = construct_player_career_records(player_ids, args.player_type, args.year_start, args.db_fp, ptype_outdir)
    print('\tTotal of {} ABs used in construction...'.format(n_abs))

    print('done :)')
