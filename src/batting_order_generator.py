__author__ = 'Connor Heaton'

import json
import time
import torch
from datetime import datetime


def find_player_recent_abs(player_id, game_pk, player_career_data, form_n_abs):
    recent_ab_fps = []

    player_all_games = player_career_data[player_id]['game_ids']
    player_game_abs = player_career_data[player_id]['game_ab_fps']

    player_career_idx = player_all_games.index(game_pk)
    curr_player_game_idx = player_career_idx - 1
    while curr_player_game_idx > 0 and len(recent_ab_fps) < form_n_abs:
        curr_game_ab_fps = player_game_abs[curr_player_game_idx]
        for game_ab_fp in curr_game_ab_fps[::-1]:
            if len(recent_ab_fps) < form_n_abs:
                recent_ab_fps.insert(0, game_ab_fp)

        curr_player_game_idx -= 1

    return recent_ab_fps


def format_batting_order_data(game_pk, batting_order, player_career_data, dataset, form_n_abs):
    data_keys = None
    batting_order_data = []

    for player_id in batting_order:
        player_recent_abs = find_player_recent_abs(player_id, game_pk, player_career_data, form_n_abs)
        player_inputs = dataset.parse_player_file_set(player_recent_abs)
        if data_keys is None:
            data_keys = player_inputs.keys()

        player_inputs = dataset.distort_item(player_inputs)
        player_inputs = {k: v[0, :].unsqueeze(0) for k, v in player_inputs.items()}
        batting_order_data.append(player_inputs)

    new_batting_order_data = {}
    for k in data_keys:
        k_data = torch.cat([bod[k].unsqueeze(0) for bod in batting_order_data], dim=0)
        new_batting_order_data[k] = k_data

    return new_batting_order_data


class BattingOrderGenerator(object):
    def __init__(self, player_career_data, dataset, form_n_abs, idx, in_q, out_q, player_type,
                 term_item='[TERM]'):
        self.player_career_data = player_career_data
        self.dataset = dataset
        self.form_n_abs = form_n_abs
        self.idx = idx
        self.in_q = in_q
        self.out_q = out_q
        self.player_type = player_type
        self.term_item = term_item

    def work(self):
        print('BattingOrderGenerator {} starting...'.format(self.idx))
        n_games = 0
        worker_start_time = time.time()

        while True:
            if self.in_q.empty():
                print('[{0}] Worker {1} sleeping for input data...'.format(
                    datetime.now().strftime("%H:%M:%S"), self.idx))
                time.sleep(2)
            else:
                item = self.in_q.get()
                if item == self.term_item:
                    print('[{0}] Worker {1} rcvd term item... adding to out_q...'.format(
                        datetime.now().strftime("%H:%M:%S"), self.idx))
                    self.out_q.put(self.term_item)
                    break
                else:
                    game_pk, game_fp = item
                    game_j = json.load(open(game_fp))
                    if self.player_type == 'batter':
                        away_player_order = game_j['away_batting_order']
                        home_player_order = game_j['home_batting_order']
                    else:
                        away_player_order = [game_j['away_starter']['__id__']]
                        home_player_order = [game_j['home_starter']['__id__']]

                    all_player_ids = away_player_order[:]
                    all_player_ids.extend(home_player_order)
                    away_batting_order_inputs = format_batting_order_data(game_pk, away_player_order,
                                                                          self.player_career_data,
                                                                          self.dataset, self.form_n_abs)
                    home_batting_order_inputs = format_batting_order_data(game_pk, home_player_order,
                                                                          self.player_career_data,
                                                                          self.dataset, self.form_n_abs)
                    agg_inputs = {k: torch.cat([away_batting_order_inputs[k], home_batting_order_inputs[k]], dim=0)
                                  for k in away_batting_order_inputs.keys()}
                    self.out_q.put([game_pk, all_player_ids, agg_inputs])
                    n_games += 1
                    if n_games % 100 == 0:
                        print('[{0}] Worker {1} has processed {2} items ({3:.2}s/item)'.format(
                            datetime.now().strftime("%H:%M:%S"), self.idx, n_games,
                            (time.time() - worker_start_time) / n_games
                        ))

        print('*** Generator {} exiting after processing {} games ***'.format(self.idx, n_games))
