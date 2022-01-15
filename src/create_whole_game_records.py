__author__ = 'Connor Heaton'

import os
import time
import sqlite3
import argparse

import numpy as np

from datetime import date, datetime
from multiprocessing import Process, Manager

from WholeGameWriter import WholeGameWriter
from WholeGameConstructor import WholeGameConstructor


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


def get_game_ids(fp, table_name, reverse_pk_order=False, start_year=2015, end_year=2019):
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

    query_args = (start_year, end_year)
    conn = sqlite3.connect(fp)
    c = conn.cursor()
    c.execute(query, query_args)
    rows = c.fetchall()

    game_ids = [r[0] for r in rows]

    return game_ids


def create_whole_game_records(args):
    n_workers = args.n_workers
    term_item = args.term_item

    print('Getting all game pks...')
    all_game_pks = get_game_ids(args.db_fp, args.games_table, reverse_pk_order=args.reverse_pk_order,
                                start_year=args.start_year, end_year=args.end_year)
    print('\tFound {} pks...'.format(len(all_game_pks)))
    m = Manager()
    write_q = m.Queue()
    worker_qs = [m.Queue() for _ in range(n_workers)]
    constructors = [WholeGameConstructor(args=args, idx=i,
                                         in_q=q, out_q=write_q) for i, q in enumerate(worker_qs)]
    writer = WholeGameWriter(args=args, in_q=write_q)
    constructor_threads = [Process(target=c.construct_whole_games, args=()) for c in constructors]
    writer_thread = Process(target=writer.write_whole_games, args=())

    print('Starting {} WholeGameConstructor processes...'.format(len(constructor_threads)))
    for t in constructor_threads:
        t.start()

    print('Starting WholeGameWriter process...')
    writer_thread.start()
    summary_every = 50
    for game_pk_idx, game_pk in enumerate(all_game_pks):
        if int(game_pk) in args.bad_game_pks:
            print_str = '* Orchestrator found bad game PK: {} *'.format(game_pk)
            print('*' * len(print_str))
            print(print_str)
            print('*' * len(print_str))

            write_q.put('[BAD_DATA]')
        else:
            pushed_to_q = False

            while not pushed_to_q:
                for worker_q in worker_qs:
                    if worker_q.empty():
                        worker_q.put(game_pk)
                        pushed_to_q = True
                        break
                # if not pushed_to_q:
                #     print('** Orchestrator sleeping **')
                #     time.sleep(0.01)

            if game_pk_idx % summary_every == 0:
                print('[{0}] Orchestrator pushed {1} of {2} game pks ({3:.2f}%) to q...'.format(
                    datetime.now().strftime("%H:%M:%S"), game_pk_idx,
                    len(all_game_pks), game_pk_idx / len(all_game_pks) * 100))

    print('Orchestrator pushed all games to qs... Pushing term item...')
    for q in worker_qs:
        q.put(term_item)

    sleep_time = 10
    while writer_thread.is_alive():
        print('Writer thread is still alive... sleeping for {} seconds...'.format(sleep_time))
        time.sleep(sleep_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year', default=2015, type=int)
    parser.add_argument('--end_year', default=2019, type=int)
    parser.add_argument('--data', default='/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1')
    parser.add_argument('--out', default='/home/czh/sata1/learning_player_form/whole_game_records/by_season')
    parser.add_argument('--db_fp', default='../database/mlb.db')
    parser.add_argument('--reverse_pk_order', default=False, type=str2bool)

    parser.add_argument('--games_table', default='statcast')
    parser.add_argument('--games_col_names_fp', default='../config/statcast_column_names.txt')
    parser.add_argument('--statcast_id_to_bio_info_fp', default='../config/statcast_id_to_bio_info.json')
    parser.add_argument('--player_id_map_fp', default='../config/all_player_id_mapping.json')
    parser.add_argument('--player_bio_info_fp', default='../config/statcast_id_to_bio_info.json')
    parser.add_argument('--mlb_pos_id_map_fp', default='../config/mlb_pos_mapping.json')
    parser.add_argument('--cbs_pos_id_map_fp', default='../config/cbs_pos_mapping.json')
    parser.add_argument('--espn_pos_id_map_fp', default='../config/espn_pos_mapping.json')

    parser.add_argument('--summary_every_n_games_constructor', default=200, type=int)
    parser.add_argument('--summary_every_n_games_writer', default=100, type=int)

    parser.add_argument('--n_workers', default=3, type=int)
    parser.add_argument('--term_item', default='<END>')
    parser.add_argument('--verbosity', default=0, type=int)

    parser.add_argument('--bad_data_fps', default=['2010/263834-10.json', '2010/263906-47.json', '2010/264008-52.json',
                                                   '2010/264107-5.json', '2010/264292-64.json', '2010/264571-18.json',
                                                   '2010/264586-35.json', '2010/264385-42.json', '2010/264704-11.json',
                                                   '2010/264957-62.json', '2011/286984-44.json', '2011/287207-69.json',
                                                   '2011/287314-29.json', '2011/287495-47.json', '2011/287913-71.json',
                                                   '2011/288355-52.json', '2012/318485-54.json', '2012/319562-64.json',
                                                   '2012/319839-53.json', '2012/319985-52.json', '2013/346964-82.json',
                                                   '2013/347543-40.json', '2014/380894-26.json', '2014/381361-49.json',
                                                   '2015/413849-40.json', '2015/414020-81.json', '2015/414264-19.json',
                                                   '2015/414292-41.json', '2015/415513-4.json', '2015/415933-70.json',
                                                   '2017/492054-24.json', '2018/529812-66.json', '2018/530969-66.json',
                                                   '2011/388034-58.json', '2011/288061-17.json', '2011/288872-72.json',
                                                   '2011/288804-72.json', '2011/288804-73.json', '2011/287987-87.json',
                                                   '2011/288685-50.json', '2011/288685-51.json', '2011/286969-37.json',
                                                   '2011/288000-67.json', '2011/288000-68.json', '2010/263834-10.json',
                                                   '2010/263906-47.json', '2010/264292-64.json', '2010/264586-35.json',
                                                   '2010/264385-42.json', '2010/264704-11.json', '2010/264957-62.json',
                                                   '2011/287314-29.json', '2011/287495-47.json', '2011/287913-71.json',
                                                   '2011/288355-52.json', '2011/288614-59.json', '2012/318485-54.json',
                                                   '2012/319562-64.json', '2012/319839-53.json', '2012/319985-52.json',
                                                   '2013/346964-82.json', '2013/347543-40.json', '2014/380894-26.json',
                                                   '2015/414020-81.json', '2015/414264-19.json', '2015/415933-70.json',
                                                   '2017/492054-24.json', '2018/529812-66.json', '2011/288034-58.json',
                                                   '2011/286984-44.json', '2011/287207-69.json', '2019/567172-14.json'
                                                   ], type=str, nargs='+',
                        help='FPs w/ corrupted data (likely b/c statcast)')
    args = parser.parse_args()
    args.bad_game_pks = [int(os.path.basename(bdf).split('-')[0]) for bdf in args.bad_data_fps]
    args.bad_game_pks = list(np.unique(args.bad_game_pks))

    create_whole_game_records(args)