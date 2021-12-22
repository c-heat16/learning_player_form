__author__ = 'Connor Heaton'

import os
import json
import time
import sqlite3


from datetime import date, datetime


def try_cast_int(x):
    try:
        x = int(x)
    except:
        pass

    return x


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {try_cast_int(k): v for k, v in x.items()}
    return x


def corrupted_pitch_order(p_list):
    corrupt = False
    last_pitch_no = 0
    p_numbers = [p['pitch_number'] for p in p_list]
    for p_no in p_numbers:
        diff = p_no - last_pitch_no
        last_pitch_no = p_no
        if diff != 1:
            corrupt = True
            break

    return corrupt


class WholeGameConstructor(object):
    def __init__(self, args, idx, in_q, out_q):
        self.args = args
        self.idx = idx
        self.in_q = in_q
        self.out_q = out_q

        self.player_id_map_fp = getattr(self.args, 'player_id_map_fp', '../config/all_player_id_mapping.json')
        self.player_id_map = json.load(open(self.player_id_map_fp), object_hook=jsonKeys2int)
        self.player_bio_info_fp = getattr(self.args, 'player_bio_info_fp',
                                          '../config/statcast_id_to_bio_info.json')
        self.player_bio_info_mapping = json.load(open(self.player_bio_info_fp), object_hook=jsonKeys2int)
        self.mlb_pos_map = json.load(open(args.mlb_pos_id_map_fp))
        self.cbs_pos_map = json.load(open(args.cbs_pos_id_map_fp))
        self.espn_pos_map = json.load(open(args.espn_pos_id_map_fp))

        self.statcast_id_to_bio_info_fp = self.args.statcast_id_to_bio_info_fp
        self.summary_every_n_games = self.args.summary_every_n_games_constructor
        self.data_basedir = self.args.data
        self.verbosity = self.args.verbosity
        self.statcast_id_to_bio_info = json.load(open(self.statcast_id_to_bio_info_fp))
        self.term_item = self.args.term_item
        self.n_workers = self.args.n_workers
        self.db_fp = self.args.db_fp
        self.games_table = self.args.games_table
        self.possible_seasons = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    def construct_whole_games(self):
        start_time = time.time()
        n_games = 0
        game_pk_idx = 0

        while True:
            if self.in_q.empty():
                print('[{}] Worker {} sleeping b/c in_q empty...'.format(datetime.now().strftime("%H:%M:%S"),
                                                                         self.idx))
                time.sleep(2)
            else:
                game_pk = self.in_q.get()

                if game_pk == self.term_item:
                    print('[{0}] Worker {1} rcvd term item... adding to out_q...'.format(
                        datetime.now().strftime("%H:%M:%S"), self.idx))
                    self.out_q.put(self.term_item)
                    break
                else:
                    whole_game_record = self.construct_whole_game(game_pk)
                    if whole_game_record is not None:
                        n_games += 1
                        self.out_q.put(whole_game_record)

                        if game_pk_idx % self.summary_every_n_games == 0:
                            now = datetime.now()
                            current_time = now.strftime("%H:%M:%S")
                            elapsed_time = time.time() - start_time
                            avg_time_per_game = elapsed_time / n_games

                            print('[{0}] Worker: {1} G: {2} s/game: {3:.2f}s'.format(current_time, self.idx,
                                                                                     game_pk_idx, avg_time_per_game))

                        game_pk_idx += 1
                    else:
                        self.out_q.put('[BAD_DATA]')

        print('*** Worker {} processed all game_pks ***'.format(self.idx))

    def query_db(self, query, args=None):
        conn = sqlite3.connect(self.db_fp, check_same_thread=False)
        c = conn.cursor()
        if args is None:
            c.execute(query)
        else:
            c.execute(query, args)
        rows = c.fetchall()

        return rows

    def get_game_meta_info(self, game_pk):
        query_str = """SELECT home_team, away_team, post_home_score, post_away_score, inning_topbot
                       from statcast
                       where game_pk = ?
                       order by at_bat_number desc, pitch_number desc;"""
        query_args = (game_pk,)

        meta_info = self.query_db(query_str, query_args)[0]

        home_score = meta_info[2]
        away_score = meta_info[3]
        inning_topbot = meta_info[4].lower()
        if home_score == away_score:
            if inning_topbot == 'bot':
                home_score += 1
            else:
                away_score += 1

        meta_d = {'home_team': meta_info[0], 'away_team': meta_info[1],
                  'home_score': home_score, 'away_score': away_score}
        return meta_d

    def add_player_bio_info(self, player_j):
        player_id = player_j['__id__']
        bio_info = self.player_bio_info_mapping[player_id]

        player_bio = {'key_mlbam': bio_info['key_mlbam'],
                      'key_retro': bio_info['key_retro'],
                      'key_bbref': bio_info['key_bbref'],
                      'key_fangraphs': bio_info['key_fangraphs'],
                      'mlb_played_first': bio_info['mlb_played_first'],
                      'mlb_played_last': bio_info['mlb_played_last'],
                      'fangraphs_name': bio_info['fangraphs_name'],
                      'mlb_pos': bio_info['mlb_pos'],
                      'mlb_pos_id': self.mlb_pos_map[bio_info['mlb_pos']],
                      'cbs_pos': bio_info['cbs_pos'],
                      'cbs_pos_id': self.cbs_pos_map[bio_info['cbs_pos']] if not str(bio_info['cbs_pos']) == 'nan' else
                      self.cbs_pos_map['NaN'],
                      'espn_pos': bio_info['espn_pos'],
                      'espn_pos_id': self.espn_pos_map[bio_info['espn_pos']] if not str(
                          bio_info['espn_pos']) == 'nan' else self.espn_pos_map['NaN'],
                      'custom_id': self.player_id_map[player_id]
                      }

        player_j['__bio__'] = player_bio
        return player_j

    def construct_whole_game(self, game_pk):
        game_j = {'home_team': None, 'away_team': None, 'home_score': None, 'away_score': None, 'game_pk': game_pk,
                  'home_batting_order': [], 'away_batting_order': [], 'home_starter': None, 'away_starter': None,
                  'home_batters': {}, 'away_batters': {}, 'matchup_data': {},
                  'ptb': {'top': {'h': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'ibb': 0},
                          'bot': {'h': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'ibb': 0}},
                  'ptb_starters': {'top': {'h': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'ibb': 0},
                                   'bot': {'h': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'ibb': 0}}}

        meta_info = self.get_game_meta_info(game_pk)
        if self.verbosity >= 2:
            print_str = '[{0}] Worker {1} meta_info: {2}'.format(datetime.now().strftime("%H:%M:%S"), self.idx,
                                                                 meta_info)
            print(print_str)

        for k, v in meta_info.items():
            game_j[k] = v

        game_season = None
        for season in self.possible_seasons:
            if os.path.exists(os.path.join(self.data_basedir, str(season), '{}-1.json'.format(game_pk))):
                game_season = season

        game_j['season'] = game_season

        n_misses_allowed = 10
        n_misses = 0
        ab_fp_tmplt = os.path.join(self.data_basedir, str(game_season), '{}-{}.json')
        curr_ab_no = 1
        while n_misses < n_misses_allowed:
            ab_fp = ab_fp_tmplt.format(game_pk, curr_ab_no)
            if self.verbosity >= 2:
                print_str = '[{0}] Worker {1} ab_fp: {2}'.format(datetime.now().strftime("%H:%M:%S"), self.idx, ab_fp)
                print(print_str)

            if os.path.exists(ab_fp):
                if self.verbosity >= 2:
                    print_str = '[{0}] Worker {1} ab_fp exists'.format(datetime.now().strftime("%H:%M:%S"), self.idx)
                    print(print_str)

                try:
                    ab_j = json.load(open(ab_fp))
                except Exception as ex:
                    print('*** Error trying to load {} ***'.format(ab_fp))
                    print('ex: {}'.format(ex))

                half_inning = ab_j['game']['inning_topbot'].lower()
                pitcher_id = ab_j['pitcher']['__id__']
                batter_id = ab_j['batter']['__id__']
                matchup_key = '{}-{}'.format(pitcher_id, batter_id)
                pitches = ab_j['pitches']

                ptb_d = {'h': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'ibb': 0}
                pitch_info = [[p['events'], p['description']] for p in ab_j['pitches']]
                for event, description in pitch_info:
                    if event == 'home_run':
                        ptb_d['hr'] += 1
                        ptb_d['h'] += 1
                    elif event in ['single', 'double', 'triple', 'home_run']:
                        ptb_d['h'] += 1
                    elif event == 'intent_walk':
                        ptb_d['ibb'] += 1
                    elif event == 'walk' and description == 'hit_by_pitch':
                        ptb_d['hbp'] += 1
                    elif event == 'walk':
                        ptb_d['bb'] += 1

                for k, v in ptb_d.items():
                    game_j['ptb'][half_inning][k] += v

                if game_j['matchup_data'].get(matchup_key, None) is None:
                    game_j['matchup_data'][matchup_key] = {k: v for k, v in ab_j['matchup'].items() if k != 'this_game'}

                if half_inning == 'top':
                    if game_j['home_starter'] is None:
                        player_j = {k: v for k, v in ab_j['pitcher'].items() if k != 'this_game'}
                        player_j = self.add_player_bio_info(player_j)
                        game_j['home_starter'] = player_j

                    if batter_id not in game_j['away_batting_order'] and len(game_j['away_batting_order']) < 9:
                        game_j['away_batting_order'].append(batter_id)
                        player_j = {k: v for k, v in ab_j['batter'].items() if k != 'this_game'}
                        player_j = self.add_player_bio_info(player_j)
                        game_j['away_batters'][batter_id] = player_j

                elif half_inning == 'bot':
                    if game_j['away_starter'] is None:
                        player_j = {k: v for k, v in ab_j['pitcher'].items() if k != 'this_game'}
                        player_j = self.add_player_bio_info(player_j)
                        game_j['away_starter'] = player_j

                    if batter_id not in game_j['home_batting_order'] and len(game_j['home_batting_order']) < 9:
                        game_j['home_batting_order'].append(batter_id)
                        player_j = {k: v for k, v in ab_j['batter'].items() if k != 'this_game'}
                        player_j = self.add_player_bio_info(player_j)
                        game_j['home_batters'][batter_id] = player_j

                if (game_j['home_starter'] is not None and game_j['home_starter']['__id__'] == int(pitcher_id)) or \
                        (game_j['away_starter'] is not None and game_j['away_starter']['__id__'] == int(pitcher_id)):

                    ptb_d = {'h': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'ibb': 0}
                    pitch_info = [[p['events'], p['description']] for p in ab_j['pitches']]
                    for event, description in pitch_info:
                        if event == 'home_run':
                            ptb_d['hr'] += 1
                            ptb_d['h'] += 1
                        elif event in ['single', 'double', 'triple', 'home_run']:
                            ptb_d['h'] += 1
                        elif event == 'intent_walk':
                            ptb_d['ibb'] += 1
                        elif event == 'walk' and description == 'hit_by_pitch':
                            ptb_d['hbp'] += 1
                        elif event == 'walk':
                            ptb_d['bb'] += 1

                    for k, v in ptb_d.items():
                        game_j['ptb_starters'][half_inning][k] += v
            else:
                if self.verbosity >= 2:
                    print_str = '[{0}] Worker {1} ab_fp does not exist'.format(datetime.now().strftime("%H:%M:%S"),
                                                                               self.idx)
                    print(print_str)

                n_misses += 1

            curr_ab_no += 1
        if self.verbosity >= 1:
            print_str = '[{0}] Worker {1} returning item from construct_whole_game...'.format(
                datetime.now().strftime("%H:%M:%S"),
                self.idx)
            print(print_str)

        return game_j
