__author__ = 'Connor Heaton'

import os
import json
import time
import numpy as np
import pandas as pd

from datetime import date, datetime


class WholeGameWriter(object):
    def __init__(self, args, in_q):
        self.args = args
        self.in_q = in_q

        self.summary_every_n_games = self.args.summary_every_n_games_writer
        self.term_item = self.args.term_item
        self.n_workers = self.args.n_workers
        self.verbosity = self.args.verbosity

        self.out_dir = self.args.out
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.season_fp_tmplt = os.path.join(self.out_dir, '{}')
        self.n_sleep = 2
        self.print_sleep_every = 2
        print_str = '[{0}] WholeGameWriter season_fp_tmplt: {1}'.format(datetime.now().strftime("%H:%M:%S"),
                                                                        self.season_fp_tmplt)
        print(print_str)

    def write_whole_games(self):
        print('Whole game writer sleeping before starting...')
        time.sleep(5)
        n_term_rcvd = 0

        n_games = 0
        game_times = []
        n_sleep = 0
        n_bad_data = 0

        while True:
            if self.in_q.empty():
                time.sleep(self.n_sleep)
                n_sleep += 1
                if n_sleep % self.print_sleep_every == 0:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")

                    print('[{}] WholeGameWriter in q empty... sleeping for {} seconds...'.format(current_time,
                                                                                                 self.n_sleep))
                time.sleep(self.n_sleep)
            else:
                n_sleep = 0
                game_start_time = time.time()
                in_data = self.in_q.get()

                if in_data == self.term_item:
                    n_term_rcvd += 1
                    print('* WholeGameWriter received term signal (total={}, max={}) *'.format(n_term_rcvd,
                                                                                               self.n_workers))
                    if n_term_rcvd == self.n_workers:
                        print('* WholeGameWriter received all term signals (n={})... stopping *'.format(n_term_rcvd))
                        break
                elif in_data == '[BAD_DATA]':
                    n_bad_data += 1
                else:
                    n_games += 1
                    game_pk = in_data['game_pk']
                    season_dir = self.season_fp_tmplt.format(in_data['season'])
                    if not os.path.exists(season_dir):
                        os.makedirs(season_dir)
                    out_fp = os.path.join(season_dir, '{}.json'.format(game_pk))
                    with open(out_fp, 'w+') as f:
                        f.write(json.dumps(in_data, indent=2))

                    game_elapsed_time = time.time() - game_start_time
                    game_times.append(game_elapsed_time)

                    if n_games % self.summary_every_n_games == 0:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print('[{}] WholeGameWriter has written {} games'.format(current_time, n_games))
                        print('\t\tAvg s/game: {0:.4f}s'.format(np.mean(game_times)))

        print('*** WholeGameWriter Summary ***')
        print('WholeGameWriter has written {} games'.format(n_games))
        print('\tn games w/ bad data: {}'.format(n_bad_data))
        print('\tAvg s/game: {0:.4f}s'.format(np.mean(game_times)))
