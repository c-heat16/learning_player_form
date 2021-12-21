__author__ = 'Connor Heaton'

import os.path
import time
import sqlite3
import argparse

from multiprocessing import Process, Manager


from AtBatWriter import AtBatWriter
from AtBatConstructor import AtBatConstructor


def str2bool(v):
    """
    Try to convert a string value to a boolean
    :param v: string value to be parsed
    :return: boolean value of string
    """
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


def get_game_ids(fp, table_name, reverse_pk_order=False, start_year=2015, end_year=2019):
    """
    Get game IDs between a given start year and end year, both inclusive
    :param fp: filepath for the SQLite3 database
    :param table_name: name of table to query
    :param reverse_pk_order: boolean indication if records should be returned in reverse order
    :param start_year: start of year range to query, inclusive
    :param end_year: end of year range to query, inclusive
    :return:
    """
    if reverse_pk_order:
        query = """SELECT DISTINCT game_pk 
                   FROM {}
                   where game_year >= ? and game_year <= ?
                   ORDER BY game_pk desc""".format(table_name)
    else:
        query = """SELECT DISTINCT game_pk 
                   FROM {}
                   where game_year >= ? and game_year <= ?
                   ORDER BY game_pk""".format(table_name)

    conn = sqlite3.connect(fp)
    c = conn.cursor()
    c.execute(query, (start_year, end_year))
    rows = c.fetchall()

    game_ids = [r[0] for r in rows]

    return game_ids


def construct_at_bat_sequences(args):
    n_workers = args.n_workers
    db_fp = args.db_fp
    games_table = args.games_table
    term_item = args.term_item
    reverse_pk_order = args.reverse_pk_order

    print('Finding all game pk\'s...')
    all_game_pks = get_game_ids(db_fp, games_table, reverse_pk_order=reverse_pk_order,
                                start_year=args.start_year, end_year=args.end_year)
    print('\tlen(all_game_pks): {}'.format(len(all_game_pks)))

    m = Manager()
    ab_q = m.Queue()
    worker_qs = [m.Queue() for _ in range(n_workers)]

    print('len(worker_pks): {}'.format(len(worker_qs)))

    constructors = [AtBatConstructor(args=args, idx=i,
                                     in_q=q, out_q=ab_q) for i, q in enumerate(worker_qs)]
    writer = AtBatWriter(args=args, in_q=ab_q)

    constructor_threads = [Process(target=c.construct_at_bats, args=()) for c in constructors]
    writer_thread = Process(target=writer.write_at_bats, args=())

    print('Starting {} AtBatConstructor processes...'.format(len(constructor_threads)))
    for t in constructor_threads:
        t.start()

    print('Starting AtBatWriter process...')
    writer_thread.start()
    summary_every = 50
    for game_pk_idx, game_pk in enumerate(all_game_pks):
        pushed_to_q = False

        while not pushed_to_q:
            for worker_q in worker_qs:
                if worker_q.empty():
                    worker_q.put(game_pk)
                    pushed_to_q = True
                    break
            if not pushed_to_q:
                time.sleep(0.1)

        if game_pk_idx % summary_every == 0:
            print('Orchestrator pushed {0} of {1} game pks ({2:.2f}%) to q...'.format(game_pk_idx,
                                                                                      len(all_game_pks),
                                                                                      game_pk_idx / len(all_game_pks) * 100))

    print('Orchestrator pushed all games to qs... Pushing term item...')
    for q in worker_qs:
        q.put(term_item)

    sleep_time = 5
    while writer_thread.is_alive():
        print('Writer thread is still alive... sleeping for {} seconds...'.format(sleep_time))
        time.sleep(sleep_time)


def read_attrs_from_file(fp):
    data = []
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '':
                data.append(line)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year', default=2015, type=int)
    parser.add_argument('--end_year', default=2019, type=int)
    parser.add_argument('--out', default='/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1')
    parser.add_argument('--db_fp', default='../database/mlb.db')
    parser.add_argument('--reverse_pk_order', default=False, type=str2bool)

    parser.add_argument('--games_table', default='statcast')
    parser.add_argument('--games_col_names_fp', default='../config/statcast_column_names.txt')
    parser.add_argument('--statcast_id_to_bio_info_fp', default='../config/statcast_id_to_bio_info.json')

    parser.add_argument('--pitching_by_season_table', default='pitching_by_season')
    parser.add_argument('--pitching_by_season_col_names_fp', default='../config/pitching_stats_column_names_fmt.txt')

    parser.add_argument('--batting_by_season_table', default='batting_by_season')
    parser.add_argument('--batting_by_season_col_names_fp', default='../config/batting_stats_column_names_fmt.txt')

    parser.add_argument('--summary_every_n_games_constructor', default=20, type=int)
    parser.add_argument('--summary_every_n_games_writer', default=20, type=int)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--term_item', default='<END>')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    game_attrs = read_attrs_from_file(args.games_col_names_fp)
    pitching_by_season_attrs = read_attrs_from_file(args.pitching_by_season_col_names_fp)
    batting_by_season_attrs = read_attrs_from_file(args.batting_by_season_col_names_fp)

    args.game_attrs = game_attrs
    args.pitching_by_season_attrs = pitching_by_season_attrs
    args.batting_by_season_attrs = batting_by_season_attrs

    construct_at_bat_sequences(args)
