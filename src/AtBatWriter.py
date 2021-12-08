__author__ = 'Connor Heaton'

import os
import json
import time
import numpy as np
import pandas as pd

from datetime import date, datetime


class AtBatWriter(object):
    def __init__(self, args, in_q):
        self.args = args
        self.in_q = in_q

        self.summary_every_n_games = self.args.summary_every_n_games_writer
        self.term_item = self.args.term_item
        self.n_workers = self.args.n_workers

        self.out_dir = self.args.out
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.season_out_dir_tmplt = os.path.join(self.out_dir, '{}')
        self.out_fp_tmplt = '{}-{}.json'
        self.n_sleep = 10
        self.print_sleep_every = 2

    def write_at_bats(self):
        """
        Write at-bat records to file as they come in from self.in_q. Writes to self.out_dir, grouped by season
        :return: None
        """
        print('At bat writer sleeping before starting...')
        time.sleep(20)
        n_term_rcvd = 0

        n_games = 0
        n_at_bats = 0
        game_times = []
        n_sleep = 0

        while True:
            if self.in_q.empty():
                time.sleep(self.n_sleep)
                n_sleep += 1
                if n_sleep % self.print_sleep_every == 0:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")

                    print('[{}] AtBatWriter in q empty... sleeping for {} seconds...'.format(current_time,
                                                                                             self.n_sleep))
                time.sleep(self.n_sleep)
            else:
                n_sleep = 0
                game_start_time = time.time()
                in_data = self.in_q.get()

                if in_data == self.term_item:
                    n_term_rcvd += 1
                    print('* AtBatWriter received term signal (total={}, max={}) *'.format(n_term_rcvd,
                                                                                           self.n_workers))
                    if n_term_rcvd == self.n_workers:
                        print('* AtBatWriter received all term signals (n={})... stopping *'.format(n_term_rcvd))
                        break
                else:
                    n_games += 1
                    for at_bat in in_data:
                        season_out_dir = self.season_out_dir_tmplt.format(at_bat['game']['game_year'])
                        if not os.path.exists(season_out_dir):
                            os.makedirs(season_out_dir)

                        at_bat_number = at_bat['game']['at_bat_number']
                        game_pk = at_bat['game']['game_pk']
                        at_bat_fp = os.path.join(season_out_dir, self.out_fp_tmplt.format(game_pk, at_bat_number))

                        with open(at_bat_fp, 'w+') as f:
                            f.write(json.dumps(at_bat, indent=4))
                            n_at_bats += 1

                    game_elapsed_time = time.time() - game_start_time
                    game_times.append(game_elapsed_time)

                    if n_games % self.summary_every_n_games == 0:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print('[{}] AtBatWriter has written {} games and {} at bats'.format(current_time, n_games,
                                                                                            n_at_bats))
                        print('\t\tAvg s/game: {0:.4f}s'.format(np.mean(game_times)))
                        print('\t\tAvg s/at bat: {0:.4f}s'.format(sum(game_times) / n_at_bats))

        print('*** AtBatWriter Summary ***')
        print('AtBatWriter has written {} games and {} at bats'.format(n_games, n_at_bats))
        print('\tAvg s/game: {0:.4f}s'.format(np.mean(game_times)))
        print('\tAvg s/at bat: {0:.4f}s'.format(sum(game_times) / n_at_bats))