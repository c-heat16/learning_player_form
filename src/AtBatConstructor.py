__author__ = 'Connor Heaton'


import json
import time
import sqlite3

import numpy as np
import pandas as pd

from datetime import date, datetime


def convert_innings_pitched(x):
    x_str = str(x)

    if '.' in x_str:
        full_innings, partial_innings = map(int, x_str.split('.'))
        partial_innings = partial_innings * 1 / 3
        x_str = '{}.{}'.format(full_innings, partial_innings)
    x = float(x)

    return x


def decode_xpos(x):
    """
    Bucketize x-axis locations over plate
    :param x: real-valued position over plate in x-axis
    :return: x-axis bucket ID
    """
    if x <= -2.0:
        x = -1.99999
    elif x >= 2.0:
        x = 1.999999

    x += 2.0
    x_id = int(x // 0.5)
    return x_id


def decode_zpos(z):
    """
    Bucketize z-axis locations over plate
    :param z: real-valued position over plate in z-axis
    :return: z-axis bucket ID
    """
    if z <= 0.0:
        z = 0.00001
    elif z >= 4.5:
        z = 4.49999999

    z_id = int(z // 0.5)
    return z_id


def decode_hit_coord(x):
    if x < 0:
        x = 0
    elif x > 250:
        x = 249
    x = int(x // 25)
    return x


class AtBatConstructor(object):
    def __init__(self, args, idx, in_q, out_q):
        self.args = args
        self.idx = idx
        self.in_q = in_q
        self.out_q = out_q

        self.statcast_id_to_bio_info_fp = self.args.statcast_id_to_bio_info_fp
        self.statcast_id_to_bio_info = json.load(open(self.statcast_id_to_bio_info_fp))

        self.term_item = self.args.term_item
        self.n_workers = self.args.n_workers
        self.db_fp = self.args.db_fp
        self.games_table = self.args.games_table
        self.summary_every_n_games = self.args.summary_every_n_games_constructor
        self.game_attr_names = args.game_attrs
        self.pitcher_attr_names = args.pitching_by_season_attrs
        self.batter_attr_names = args.batting_by_season_attrs

        self.troublesome_names = {
            'aj pollock%': 'a.j. pollock',
            'b%j% upton%': 'melvin upton',
            'byungho park%': 'byung-ho park',
            'cameron perkins%': 'cam perkins',
            'carlos sanchez%': 'yolmer sanchez',
            'dan robertson%': 'daniel robertson',
            'dan vogelbach%': 'daniel vogelbach',
            'danny dorn%': 'daniel dorn',
            'dee strange-gordon%': 'dee gordon',
            'dj stewart%': 'd.j. stewart',
            'ed easley%': 'edward easley',
            'gio urshela%': 'giovanny urshela',
            'hyunjin ryu%': 'hyun-jin ryu',
            'hyunsoo kim%': 'hyun soo kim',
            'jake brigham%': 'jacob brigham',
            'jonathon niese%': 'jon niese',
            'josh smith%': 'josh a. smith',
            'jt riddle%': 'j.t. riddle',
            'jungho kang%': 'jung ho kang',
            'matt boyd%': 'matthew boyd',
            'matthew bowman%': 'matt bowman',
            'matthew joyce%': 'matt joyce',
            'michael taylor%': 'michael a. taylor',
            'mike morse%': 'michael morse',
            'nate karns%': 'nathan karns',
            'nick castellanos%': 'nicholas castellanos',
            'nick delmonico%': 'nicky delmonico',
            'phil ervin%': 'phillip ervin',
            'philip gosselin%': 'phil gosselin',
            'rafael lopez%': 'raffy lopez',
            'rey navarro%': 'reynaldo navarro',
            'robert whalen%': 'rob whalen',
            'steven tolleson%': 'steve tolleson',
            'stevie wilkerson%': 'steve wilkerson',
            'sugarray marimon%': 'sugar marimon',
            'thomas field%': 'tommy field',
            'zach granite%': 'zack granite',
            'zach wheeler%': 'zack wheeler'
        }

        self.career_cache = {'season': -1, 'pitching': {}, 'batting': {}, 'matchup': {}}

    def construct_at_bats(self):
        """
        Construct at-bats records. Will digest game PKs from self.in_q and construct at-bat records until [TERM] signal is recieved
        :return: None
        """
        start_time = time.time()
        n_games = 0
        n_abs = 0
        game_pk_idx = 0

        while True:
            if self.in_q.empty():
                print('[{}] Worker {} sleeping b/c in_q empty...'.format(datetime.now().strftime("%H:%M:%S"),
                                                                         self.idx))
                time.sleep(5)
            else:
                game_pk = self.in_q.get()

                if game_pk == self.term_item:
                    print('[{0}] Worker {1} rcvd term item... adding to out_q ...'.format(
                        datetime.now().strftime("%H:%M:%S"),
                        self.idx))
                    self.out_q.put(self.term_item)
                    break
                else:
                    # if int(game_pk) == 570334:
                    #     print('****** [{0}] Worker {1} rcvd game pk 570334 ******'.format(
                    #         datetime.now().strftime("%H:%M:%S"),
                    #         self.idx))

                    game_at_bats = self.construct_single_game_at_bats(game_pk)
                    n_games += 1
                    n_abs += len(game_at_bats)
                    self.out_q.put(game_at_bats)

                    if game_pk_idx % self.summary_every_n_games == 0:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        elapsed_time = time.time() - start_time
                        avg_time_per_game = elapsed_time / n_games
                        avg_time_per_ab = elapsed_time / n_abs
                        print('[{0}] Worker: {1} G: {2} s/game: {3:.2f}s s/ab: {4:.2f}s'.format(current_time,
                                                                                                self.idx,
                                                                                                game_pk_idx,
                                                                                                avg_time_per_game,
                                                                                                avg_time_per_ab))

                    game_pk_idx += 1

        print('*** Worker {} processed all game_pks ***'.format(self.idx))

    def query_game_data(self, game_pk):
        """
        Query all pitch-by-pitch data for a given game
        :param game_pk: ID of game to get data for
        :return: all pitch-by-pitch records for a given game
        """
        query = """SELECT * 
                   FROM statcast 
                   WHERE game_pk={} 
                   ORDER BY at_bat_number, pitch_number ASC""".format(game_pk)

        conn = sqlite3.connect(self.db_fp, check_same_thread=False)
        c = conn.cursor()
        c.execute(query)
        rows = c.fetchall()

        return rows

    def query_db(self, query, args=None):
        """
        Query the database using a given query and set of arguments
        :param query: query to execute
        :param args: arguments to use in query
        :return: result of query
        """
        conn = sqlite3.connect(self.db_fp, check_same_thread=False)
        c = conn.cursor()
        if args is None:
            c.execute(query)
        else:
            c.execute(query, args)
        rows = c.fetchall()

        return rows

    def construct_single_game_at_bats(self, game_pk):
        """
        Construct records for all of the at-bats in a given game
        :param game_pk: PK of game for which at-bats will be constructed
        :return: a list of JSON objects, each describing an at-bat in the specified game
        """
        this_game_at_bats = []  # will be populated with JSON items, each representing 1 at bat
        processed_at_bats = []  # will contain 'complete' at bats once batter and pitcher stats added
        current_at_bat = {'game': {},
                          'pitcher': {},
                          'batter': {},
                          'matchup': {},
                          'pitches': []}
        raw_game_data = self.query_game_data(game_pk)

        for pitch_event in raw_game_data:
            # input('len(pitch_event): {}'.format(len(pitch_event)))
            pe_dict = dict(zip(self.game_attr_names, pitch_event))

            if not pe_dict['at_bat_number'] == current_at_bat['game'].get('at_bat_number', -1):
                if len(current_at_bat['pitches']) > 0:
                    # tmp = 'Something something process the current at bat since we have encountered the next one'
                    this_game_at_bats.append(current_at_bat)

                current_at_bat = {'game': {},
                                  'pitcher': {},
                                  'batter': {},
                                  'matchup': {},
                                  'pitches': []}
                current_at_bat['game']['game_pk'] = game_pk
                current_at_bat['game']['home_team'] = pe_dict['home_team']
                current_at_bat['game']['away_team'] = pe_dict['away_team']
                current_at_bat['game']['at_bat_number'] = pe_dict['at_bat_number']
                current_at_bat['game']['game_date'] = pe_dict['game_date']
                current_at_bat['game']['game_type'] = pe_dict['game_type']
                current_at_bat['game']['game_year'] = pe_dict['game_year']
                current_at_bat['game']['on_3b'] = pe_dict['on_3b']
                current_at_bat['game']['on_2b'] = pe_dict['on_2b']
                current_at_bat['game']['on_1b'] = pe_dict['on_1b']
                current_at_bat['game']['outs_when_up'] = pe_dict['outs_when_up']
                current_at_bat['game']['inning'] = pe_dict['inning']
                current_at_bat['game']['inning_topbot'] = pe_dict['inning_topbot']
                current_at_bat['game']['fielder_2'] = pe_dict['fielder_2']
                current_at_bat['game']['fielder_3'] = pe_dict['fielder_3']
                current_at_bat['game']['fielder_4'] = pe_dict['fielder_4']
                current_at_bat['game']['fielder_5'] = pe_dict['fielder_5']
                current_at_bat['game']['fielder_6'] = pe_dict['fielder_6']
                current_at_bat['game']['fielder_7'] = pe_dict['fielder_7']
                current_at_bat['game']['fielder_8'] = pe_dict['fielder_8']
                current_at_bat['game']['home_score'] = pe_dict['home_score']
                current_at_bat['game']['away_score'] = pe_dict['away_score']
                current_at_bat['game']['bat_score'] = pe_dict['bat_score']
                current_at_bat['game']['fld_score'] = pe_dict['fld_score']
                current_at_bat['game']['if_fielding_alignment'] = pe_dict['if_fielding_alignment']
                current_at_bat['game']['of_fielding_alignment'] = pe_dict['of_fielding_alignment']

                current_at_bat['pitcher']['__id__'] = pe_dict['pitcher']
                current_at_bat['pitcher']['throws'] = pe_dict['p_throws']
                current_at_bat['pitcher']['first_name'] = self.statcast_id_to_bio_info[str(pe_dict['pitcher'])][
                    'name_first']
                current_at_bat['pitcher']['last_name'] = self.statcast_id_to_bio_info[str(pe_dict['pitcher'])][
                    'name_last']

                current_at_bat['batter']['__id__'] = pe_dict['batter']
                current_at_bat['batter']['stand'] = pe_dict['stand']
                current_at_bat['batter']['first_name'] = self.statcast_id_to_bio_info[str(pe_dict['batter'])][
                    'name_first']
                current_at_bat['batter']['last_name'] = self.statcast_id_to_bio_info[str(pe_dict['batter'])][
                    'name_last']

            this_pitch = {'pitch_type': pe_dict['pitch_type'],
                          'pitch_number': pe_dict['pitch_number'],
                          'pitch_name': pe_dict['pitch_name'],
                          'release_speed': pe_dict['release_speed'],
                          'release_pos_x': pe_dict['release_pos_x'],
                          'release_pos_z': pe_dict['release_pos_z'],
                          'release_pos_y': pe_dict['release_pos_y'],
                          'release_spin_rate': pe_dict['release_spin_rate'],
                          'release_extension': pe_dict['release_extension'],
                          'hc_x': pe_dict['hc_x'],
                          'hc_y': pe_dict['hc_y'],
                          'vx0': pe_dict['vx0'],
                          'vy0': pe_dict['vy0'],
                          'vz0': pe_dict['vz0'],
                          'ax': pe_dict['ax'],
                          'ay': pe_dict['ay'],
                          'az': pe_dict['az'],
                          'hit_distance_sc': pe_dict['hit_distance_sc'],
                          'launch_speed': pe_dict['launch_speed'],
                          'launch_angle': pe_dict['launch_angle'],
                          'zone': pe_dict['zone'],
                          'plate_x': pe_dict['plate_x'],
                          'plate_z': pe_dict['plate_z'],
                          'balls': pe_dict['balls'],
                          'strikes': pe_dict['strikes'],
                          'events': pe_dict['events'],
                          'description': pe_dict['description']}
            current_at_bat['pitches'].append(this_pitch)
        this_game_at_bats.append(current_at_bat)

        season_cache = {'pitching': {}, 'batting': {}, 'matchup': {}}
        for at_bat in this_game_at_bats:
            ab_year = at_bat['game']['game_year']
            ab_date = at_bat['game']['game_date']
            ab_number = at_bat['game']['at_bat_number']
            ab_pitcher_id = at_bat['pitcher']['__id__']
            ab_batter_id = at_bat['batter']['__id__']

            if not ab_year == self.career_cache['season']:
                self.career_cache = {'season': ab_year, 'pitching': {}, 'batting': {}, 'matchup': {}}

            pitcher_career_stats = self.career_cache['pitching'].get(ab_pitcher_id, None)
            if pitcher_career_stats is None:
                pitcher_career_stats = self.get_pitcher_career_stats(ab_pitcher_id, ab_year)
                self.career_cache['pitching'][ab_pitcher_id] = pitcher_career_stats

            batter_career_stats = self.career_cache['batting'].get(ab_batter_id, None)
            if batter_career_stats is None:
                batter_career_stats = self.get_batter_career_stats(ab_batter_id, ab_year)
                self.career_cache['batting'][ab_batter_id] = batter_career_stats

            matchup_key = '{}-{}'.format(ab_pitcher_id, ab_batter_id)
            matchup_career_stats = self.career_cache['matchup'].get(matchup_key, None)
            if matchup_career_stats is None:
                matchup_career_stats = self.get_matchup_stats(ab_pitcher_id, ab_batter_id, current_year=ab_year)
                self.career_cache['matchup'][matchup_key] = matchup_career_stats

            pitcher_season_stats = season_cache['pitching'].get(ab_pitcher_id, None)
            if pitcher_season_stats is None:
                pitcher_season_stats = self.get_player_season_stats(ab_pitcher_id, game_pk, ab_year,
                                                                    player_type='pitcher')
                season_cache['pitching'][ab_pitcher_id] = pitcher_season_stats

            batter_season_stats = season_cache['batting'].get(ab_batter_id, None)
            if batter_season_stats is None:
                batter_season_stats = self.get_player_season_stats(ab_batter_id, game_pk, ab_year,
                                                                   player_type='batter')
                season_cache['batting'][ab_batter_id] = batter_season_stats

            matchup_season_stats = season_cache['matchup'].get(matchup_key, None)
            if matchup_season_stats is None:
                matchup_season_stats = self.get_matchup_stats(ab_pitcher_id, ab_batter_id, current_year=ab_year,
                                                              game_pk=game_pk)
                season_cache['matchup'][matchup_key] = matchup_season_stats

            pitcher_stats_last15 = self.get_player_season_stats(ab_pitcher_id, game_pk, ab_year, player_type='pitcher',
                                                                game_date=ab_date, ab_number=ab_number, last_n_days=15)
            batter_stats_last15 = self.get_player_season_stats(ab_batter_id, game_pk, ab_year, player_type='batter',
                                                               game_date=ab_date, ab_number=ab_number, last_n_days=15)
            next_game_state = self.get_next_gamestate(game_pk, ab_number)
            matchup_this_game_stats = self.get_matchup_stats(ab_pitcher_id, ab_batter_id, current_year=ab_year,
                                                             game_pk=game_pk, last_n_days=0, ab_number=ab_number,
                                                             game_date=ab_date)
            pitcher_stats_this_game = self.get_player_season_stats(ab_pitcher_id, game_pk, ab_year,
                                                                   player_type='pitcher',
                                                                   game_date=ab_date, ab_number=ab_number,
                                                                   last_n_days=0)
            batter_stats_this_game = self.get_player_season_stats(ab_batter_id, game_pk, ab_year, player_type='batter',
                                                                  game_date=ab_date, ab_number=ab_number,
                                                                  last_n_days=0)

            at_bat['game']['next_game_state'] = next_game_state

            at_bat['pitcher']['career'] = pitcher_career_stats
            at_bat['pitcher']['season'] = pitcher_season_stats
            at_bat['pitcher']['last15'] = pitcher_stats_last15
            at_bat['pitcher']['this_game'] = pitcher_stats_this_game

            at_bat['batter']['career'] = batter_career_stats
            at_bat['batter']['season'] = batter_season_stats
            at_bat['batter']['last15'] = batter_stats_last15
            at_bat['batter']['this_game'] = batter_stats_this_game

            at_bat['matchup']['career'] = matchup_career_stats
            at_bat['matchup']['season'] = matchup_season_stats
            at_bat['matchup']['this_game'] = matchup_this_game_stats

            processed_at_bats.append(at_bat)

        return processed_at_bats

    def get_pitcher_career_stats(self, pitcher_id, ab_year):
        """
        Get career statistics for a specified pitcher
        :param pitcher_id: ID of pitcher being queried
        :param ab_year: "current" year. Statistics for years *before* ab_year will be returned
        :return: Career statistics for a specified player up until a certain year
        """
        p_bio_info = self.statcast_id_to_bio_info[str(pitcher_id)]
        p_first_played = int(p_bio_info['mlb_played_first'])
        p_last_played = int(p_bio_info['mlb_played_last'])
        p_fullname = p_bio_info['fangraphs_name']

        if p_fullname == 'UNK':
            p_fname = p_bio_info['name_first']
            p_lname = p_bio_info['name_last']

            p_fname = p_fname.replace('.', '%')
            p_fname = p_fname.replace(' ', '')
            p_fname = p_fname.replace('é', 'e')
            p_lname = p_lname.replace('.', '%')
            p_lname = p_lname.replace(' ', '')
            p_lname = p_lname.replace('é', 'e')
            p_fullname = '{} {}%'.format(p_fname, p_lname)
            raw_fullname = p_fullname[:]
            p_fullname = self.troublesome_names.get(p_fullname, p_fullname)
            if p_fullname != raw_fullname:
                print('\t!! {} mapped to {} !!'.format(raw_fullname, p_fullname))

        attr_names = ['Season', 'Team', 'Age', 'W', 'L', 'ERA', 'WAR', 'G', 'GS', 'CG', 'ShO', 'SV', 'BS', 'IP', 'TBF',
                      'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'HBP', 'WP', 'BK', 'SO', 'GB', 'FB', 'LD', 'IFFB', 'Balls',
                      'Strikes', 'Pitches', 'RS', 'AVG', 'Pulls', 'FB_pct', 'FBv', 'SL_pct', 'SLv', 'CT_pct', 'CTv',
                      'CB_pct', 'CBv', 'CH_pct', 'CHv', 'SF_pct', 'SFv', 'KN_pct', 'KNv', 'XX_pct', 'O_Swing_pct',
                      'Z_Swing_pct', 'Swing_pct', 'O_Contact_pct', 'Z_Contact_pct', 'Contact_pct', 'Zone_pct',
                      'F_Strike_pct', 'SwStr_pct', 'HLD', 'Pull_pct', 'Cent_pct', 'Oppo_pct', 'Soft_pct', 'Med_pct',
                      'Hard_pct',
                      # new for v5
                      'BABIP', 'LOB_pct', 'FIP', 'RAR', 'tERA', 'xFIP', '_WPA', 'pos_WPA', 'RE24', 'REW',
                      'pLI', 'inLI', 'gmLI', 'exLI', 'Clutch', 'SIERA', 'Pace',
                      # new for v6 (BERT-esque approach)
                      'Starting', 'Start_IP', 'Relieving', 'Relief_IP', 'wFB', 'wSL', 'wCT', 'wCB', 'wCH', 'wSF', 'wKN',

                      ]

        pitcher_career_data_query = """SELECT Season, Team, Age, W, L, ERA, WAR, G, GS, CG, ShO, SV, BS, IP, TBF,
                                              H, R, ER, HR, BB, IBB, HBP, WP, BK, SO, GB, FB, LD, IFFB,
                                              Balls, Strikes, Pitches, RS, AVG, Pulls, FB_pct, FBv, SL_pct,
                                              SLv, CT_pct, CTv, CB_pct, CBv, CH_pct, CHv, SF_pct, SFv, KN_pct,
                                              KNv, XX_pct, O_Swing_pct, Z_Swing_pct, Swing_pct, O_Contact_pct, 
                                              Z_Contact_pct, Contact_pct, Zone_pct, F_Strike_pct, SwStr_pct,
                                              HLD, Pull_pct, Cent_pct, Oppo_pct, Soft_pct, Med_pct, Hard_pct,
                                              BABIP, LOB_pct, FIP, RAR, tERA, xFIP, _WPA, pos_WPA, RE24, REW,
                                              pLI, inLI, gmLI, exLI, Clutch, SIERA, Pace,
                                              Starting, Start_IP, Relieving, Relief_IP, wFB, wSL, wCT, wCB,
                                              wCH, wSF, wKN
                                           FROM pitching_by_season
                                           WHERE LOWER(Name) LIKE ?
                                           AND Season >= ?
                                           AND Season <= ?
                                           AND Season < ?"""
        query_args = (p_fullname, p_first_played, p_last_played, ab_year)
        raw_pitcher_career_data = self.query_db(pitcher_career_data_query, query_args)
        pitcher_df = pd.DataFrame(raw_pitcher_career_data, columns=attr_names).fillna(0)

        if pitcher_df.shape[0] > 0:
            seasons_played = pitcher_df.shape[0]
            current_age = int(pitcher_df['Age'].max()) + 1

            pitcher_df['IP'].apply(convert_innings_pitched)
            pitcher_df['WHIP'] = (pitcher_df['BB'] + pitcher_df['H']) / pitcher_df['IP']

            # Reworked w/ v6
            career_ip_sum = pitcher_df['IP'].sum()
            career_ip_avg = pitcher_df['IP'].mean()
            career_tbf_sum = pitcher_df['TBF'].sum()
            career_tbf_avg = pitcher_df['TBF'].mean()
            career_pitches_sum = pitcher_df['Pitches'].sum()
            career_pitches_avg = pitcher_df['Pitches'].mean()
            career_swing_pct_sum = pitcher_df['Swing_pct'].sum()
            career_swing_pct_avg = pitcher_df['Swing_pct'].mean()
            career_contact_pct_sum = pitcher_df['Contact_pct'].sum()
            career_contact_pct_avg = pitcher_df['Contact_pct'].mean()

            # Pitch percents
            career_fb_pct = pitcher_df['FB_pct'].mean()
            career_sl_pct = pitcher_df['SL_pct'].mean()
            career_ct_pct = pitcher_df['CT_pct'].mean()
            career_cb_pct = pitcher_df['CB_pct'].mean()
            career_ch_pct = pitcher_df['CH_pct'].mean()
            career_sf_pct = pitcher_df['SF_pct'].mean()
            career_kn_pct = pitcher_df['KN_pct'].mean()
            career_xx_pct = pitcher_df['XX_pct'].mean()

            # Pitch velocities
            career_fbv = pitcher_df['FBv'].mean()
            career_slv = pitcher_df['SLv'].mean()
            career_ctv = pitcher_df['CTv'].mean()
            career_cbv = pitcher_df['CBv'].mean()
            career_chv = pitcher_df['CHv'].mean()
            career_sfv = pitcher_df['SFv'].mean()
            career_knv = pitcher_df['KNv'].mean()

            # Misc stats
            career_rs = pitcher_df['RS'].mean()
            career_avg = pitcher_df['AVG'].mean()

            career_zone_pct = pitcher_df['Zone_pct'].mean()
            career_fstrike_pct = pitcher_df['F_Strike_pct'].mean()
            career_pull_pct = pitcher_df['Pull_pct'].mean()
            career_cent_pct = pitcher_df['Cent_pct'].mean()
            career_oppo_pct = pitcher_df['Oppo_pct'].mean()
            career_soft_pct = pitcher_df['Soft_pct'].mean()
            career_med_pct = pitcher_df['Med_pct'].mean()
            career_hard_pct = pitcher_df['Hard_pct'].mean()
            career_swstr_pct = pitcher_df['SwStr_pct'].mean()

            career_o_swing_pct = pitcher_df['O_Swing_pct'].mean()
            career_z_swing_pct = pitcher_df['Z_Swing_pct'].mean()
            # career_swing_pct = pitcher_df['Swing_pct'].mean()

            career_o_contact_pct = pitcher_df['O_Contact_pct'].mean()
            career_z_contact_pct = pitcher_df['Z_Contact_pct'].mean()
            # career_contact_pct = pitcher_df['Contact_pct'].mean()

            career_h_sum = pitcher_df['H'].sum()
            career_h_avg = pitcher_df['H'].mean()
            career_bb_sum = pitcher_df['BB'].sum()
            career_bb_avg = pitcher_df['BB'].mean()
            career_wp_sum = pitcher_df['WP'].sum()
            career_wp_avg = pitcher_df['WP'].mean()

            pitcher_df['Strike_pct'] = pitcher_df['Strikes'] / pitcher_df['Pitches']
            pitcher_df['Ball_pct'] = pitcher_df['Balls'] / pitcher_df['Pitches']
            career_strike_pct = pitcher_df['Strike_pct'].mean()
            career_ball_pct = pitcher_df['Ball_pct'].mean()

            career_pulls_sum = pitcher_df['Pulls'].sum()
            career_pulls_avg = pitcher_df['Pulls'].mean()
            career_bk_sum = pitcher_df['BK'].sum()
            career_bk_avg = pitcher_df['BK'].mean()
            career_so_sum = pitcher_df['SO'].sum()
            career_so_avg = pitcher_df['SO'].mean()

            career_gb_sum = pitcher_df['GB'].sum()
            career_gb_avg = pitcher_df['GB'].mean()
            career_fb_sum = pitcher_df['FB'].sum()
            career_fb_avg = pitcher_df['FB'].mean()
            career_ld_sum = pitcher_df['LD'].sum()
            career_ld_avg = pitcher_df['LD'].mean()

            career_era = pitcher_df['ERA'].mean()
            career_hld_avg = pitcher_df['HLD'].mean()
            career_hld_sum = pitcher_df['HLD'].sum()
            career_w_avg = pitcher_df['W'].mean()
            career_w_sum = pitcher_df['W'].sum()
            career_l_avg = pitcher_df['L'].mean()
            career_l_sum = pitcher_df['L'].sum()
            career_g_avg = pitcher_df['G'].mean()
            career_g_sum = pitcher_df['G'].sum()
            career_gs_avg = pitcher_df['GS'].mean()
            career_gs_sum = pitcher_df['GS'].sum()
            career_cg_avg = pitcher_df['CG'].mean()
            career_cg_sum = pitcher_df['CG'].sum()
            career_sho_avg = pitcher_df['ShO'].mean()
            career_sho_sum = pitcher_df['ShO'].sum()
            career_sv_avg = pitcher_df['SV'].mean()
            career_sv_sum = pitcher_df['SV'].sum()
            career_bs_avg = pitcher_df['BS'].mean()
            career_bs_sum = pitcher_df['BS'].sum()

            # career_whip = (pitcher_df['BB'].sum() + pitcher_df['H'].sum()) / pitcher_df['IP'].sum()
            career_whip_avg = pitcher_df['WHIP'].mean()
            career_whip_sum = pitcher_df['WHIP'].sum()

            # v5
            career_war_avg = pitcher_df['WAR'].mean()
            career_war_sum = pitcher_df['WAR'].sum()
            career_babip_avg = pitcher_df['BABIP'].mean()
            career_babip_sum = pitcher_df['BABIP'].sum()
            career_lob_pct_avg = pitcher_df['LOB_pct'].mean()
            career_lob_pct_sum = pitcher_df['LOB_pct'].sum()
            career_fip_avg = pitcher_df['FIP'].mean()
            career_fip_sum = pitcher_df['FIP'].sum()
            career_rar_avg = pitcher_df['RAR'].mean()
            career_rar_sum = pitcher_df['RAR'].sum()
            career_tera_avg = pitcher_df['tERA'].mean()
            career_tera_sum = pitcher_df['tERA'].sum()
            career_xfip_avg = pitcher_df['xFIP'].mean()
            career_xfip_sum = pitcher_df['xFIP'].sum()
            career_neg_wpa_avg = pitcher_df['_WPA'].mean()
            career_neg_wpa_sum = pitcher_df['_WPA'].sum()
            career_pos_wpa_avg = pitcher_df['pos_WPA'].mean()
            career_pos_wpa_sum = pitcher_df['pos_WPA'].sum()
            career_re24_avg = pitcher_df['RE24'].mean()
            career_re24_sum = pitcher_df['RE24'].sum()
            career_rew_avg = pitcher_df['REW'].mean()
            career_rew_sum = pitcher_df['REW'].sum()
            career_pli_avg = pitcher_df['pLI'].mean()
            career_pli_sum = pitcher_df['pLI'].sum()
            career_inli_avg = pitcher_df['inLI'].mean()
            career_inli_sum = pitcher_df['inLI'].sum()
            career_gmli_avg = pitcher_df['gmLI'].mean()
            career_gmli_sum = pitcher_df['gmLI'].sum()
            career_exli_avg = pitcher_df['exLI'].mean()
            career_exli_sum = pitcher_df['exLI'].sum()
            career_clutch_avg = pitcher_df['Clutch'].mean()
            career_clutch_sum = pitcher_df['Clutch'].sum()
            career_siera_avg = pitcher_df['SIERA'].mean()
            career_siera_sum = pitcher_df['SIERA'].sum()
            career_pace_avg = pitcher_df['Pace'].mean()
            career_pace_sum = pitcher_df['Pace'].sum()

            # v6
            career_starting_avg = pitcher_df['Starting'].mean()
            career_starting_sum = pitcher_df['Starting'].sum()
            career_start_ip_avg = pitcher_df['Start_IP'].mean()
            career_start_ip_sum = pitcher_df['Start_IP'].sum()
            career_relieving_avg = pitcher_df['Relieving'].mean()
            career_relieving_sum = pitcher_df['Relieving'].sum()
            career_relief_ip_avg = pitcher_df['Relief_IP'].mean()
            career_relief_ip_sum = pitcher_df['Relief_IP'].sum()
            career_wfb_avg = pitcher_df['wFB'].mean()
            career_wfb_sum = pitcher_df['wFB'].sum()
            career_wsl_avg = pitcher_df['wSL'].mean()
            career_wsl_sum = pitcher_df['wSL'].sum()
            career_wct_avg = pitcher_df['wCT'].mean()
            career_wct_sum = pitcher_df['wCT'].sum()
            career_wcb_avg = pitcher_df['wCB'].mean()
            career_wcb_sum = pitcher_df['wCB'].sum()
            career_wch_avg = pitcher_df['wCH'].mean()
            career_wch_sum = pitcher_df['wCH'].sum()
            career_wsf_avg = pitcher_df['wSF'].mean()
            career_wsf_sum = pitcher_df['wSF'].sum()
            career_wkn_avg = pitcher_df['wKN'].mean()
            career_wkn_sum = pitcher_df['wKN'].sum()

            career_stats = {'seasons_played': seasons_played,
                            'current_age': current_age,
                            'career_fb_pct': career_fb_pct,
                            'career_sl_pct': career_sl_pct,
                            'career_ct_pct': career_ct_pct,
                            'career_cb_pct': career_cb_pct,
                            'career_ch_pct': career_ch_pct,
                            'career_sf_pct': career_sf_pct,
                            'career_kn_pct': career_kn_pct,
                            'career_xx_pct': career_xx_pct,
                            'career_fbv': career_fbv,
                            'career_slv': career_slv,
                            'career_ctv': career_ctv,
                            'career_cbv': career_cbv,
                            'career_chv': career_chv,
                            'career_sfv': career_sfv,
                            'career_knv': career_knv,
                            'career_opp_avg': career_avg,
                            # 'career_wp_pct': 0,
                            'career_strike_pct': career_strike_pct,
                            'career_ball_pct': career_ball_pct,
                            'career_zone_pct': career_zone_pct,
                            'career_swstr_pct': career_swstr_pct,
                            'career_pull_pct': career_pull_pct,
                            'career_cent_pct': career_cent_pct,
                            'career_oppo_pct': career_oppo_pct,
                            'career_soft_pct': career_soft_pct,
                            'career_med_pct': career_med_pct,
                            'career_hard_pct': career_hard_pct,
                            'career_f_strike_pct': career_fstrike_pct,
                            'career_era': career_era,
                            'career_rs': career_rs,
                            'career_o_swing_pct': career_o_swing_pct,
                            'career_z_swing_pct': career_z_swing_pct,
                            'career_o_contact_pct': career_o_contact_pct,
                            'career_z_contact_pct': career_z_contact_pct,
                            # ADDED V5
                            'career_war_avg': career_war_avg,
                            'career_war_sum': career_war_sum,
                            # ADDED V6
                            'career_starting_avg': career_starting_avg,
                            'career_starting_sum': career_starting_sum,
                            'career_start_ip_avg': career_start_ip_avg,
                            'career_start_ip_sum': career_start_ip_sum,
                            'career_relieving_avg': career_relieving_avg,
                            'career_relieving_sum': career_relieving_sum,
                            'career_relief_ip_avg': career_relief_ip_avg,
                            'career_relief_ip_sum': career_relief_ip_sum,
                            'career_wfb_avg': career_wfb_avg,
                            'career_wfb_sum': career_wfb_sum,
                            'career_wsl_avg': career_wsl_avg,
                            'career_wsl_sum': career_wsl_sum,
                            'career_wct_avg': career_wct_avg,
                            'career_wct_sum': career_wct_sum,
                            'career_wcb_avg': career_wcb_avg,
                            'career_wcb_sum': career_wcb_sum,
                            'career_wch_avg': career_wch_avg,
                            'career_wch_sum': career_wch_sum,
                            'career_wsf_avg': career_wsf_avg,
                            'career_wsf_sum': career_wsf_sum,
                            'career_wkn_avg': career_wkn_avg,
                            'career_wkn_sum': career_wkn_sum,
                            # REWORKED
                            'career_ip_sum': career_ip_sum,
                            'career_ip_avg': career_ip_avg,
                            'career_tbf_sum': career_tbf_sum,
                            'career_tbf_avg': career_tbf_avg,
                            'career_pitches_sum': career_pitches_sum,
                            'career_pitches_avg': career_pitches_avg,
                            'career_swing_pct_sum': career_swing_pct_sum,
                            'career_swing_pct_avg': career_swing_pct_avg,
                            'career_contact_pct_sum': career_contact_pct_sum,
                            'career_contact_pct_avg': career_contact_pct_avg,
                            'career_h_sum': career_h_sum,
                            'career_h_avg': career_h_avg,
                            'career_bb_sum': career_bb_sum,
                            'career_bb_avg': career_bb_avg,
                            'career_wp_sum': career_wp_sum,
                            'career_wp_avg': career_wp_avg,
                            'career_pulls_sum': career_pulls_sum,
                            'career_pulls_avg': career_pulls_avg,
                            'career_bk_sum': career_bk_sum,
                            'career_bk_avg': career_bk_avg,
                            'career_so_sum': career_so_sum,
                            'career_so_avg': career_so_avg,
                            'career_gb_sum': career_gb_sum,
                            'career_gb_avg': career_gb_avg,
                            'career_fb_sum': career_fb_sum,
                            'career_fb_avg': career_fb_avg,
                            'career_ld_sum': career_ld_sum,
                            'career_ld_avg': career_ld_avg,
                            'career_hld_avg': career_hld_avg,
                            'career_hld_sum': career_hld_sum,
                            'career_w_avg': career_w_avg,
                            'career_w_sum': career_w_sum,
                            'career_l_avg': career_l_avg,
                            'career_l_sum': career_l_sum,
                            'career_g_avg': career_g_avg,
                            'career_g_sum': career_g_sum,
                            'career_gs_avg': career_gs_avg,
                            'career_gs_sum': career_gs_sum,
                            'career_cg_avg': career_cg_avg,
                            'career_cg_sum': career_cg_sum,
                            'career_sho_avg': career_sho_avg,
                            'career_sho_sum': career_sho_sum,
                            'career_sv_avg': career_sv_avg,
                            'career_sv_sum': career_sv_sum,
                            'career_bs_avg': career_bs_avg,
                            'career_bs_sum': career_bs_sum,
                            'career_whip_avg': career_whip_avg,
                            'career_whip_sum': career_whip_sum,
                            'career_babip_avg': career_babip_avg,
                            'career_babip_sum': career_babip_sum,
                            'career_lob_pct_avg': career_lob_pct_avg,
                            'career_lob_pct_sum': career_lob_pct_sum,
                            'career_fip_avg': career_fip_avg,
                            'career_fip_sum': career_fip_sum,
                            'career_rar_avg': career_rar_avg,
                            'career_rar_sum': career_rar_sum,
                            'career_tera_avg': career_tera_avg,
                            'career_tera_sum': career_tera_sum,
                            'career_xfip_avg': career_xfip_avg,
                            'career_xfip_sum': career_xfip_sum,
                            'career_neg_wpa_avg': career_neg_wpa_avg,
                            'career_neg_wpa_sum': career_neg_wpa_sum,
                            'career_pos_wpa_avg': career_pos_wpa_avg,
                            'career_pos_wpa_sum': career_pos_wpa_sum,
                            'career_re24_avg': career_re24_avg,
                            'career_re24_sum': career_re24_sum,
                            'career_rew_avg': career_rew_avg,
                            'career_rew_sum': career_rew_sum,
                            'career_pli_avg': career_pli_avg,
                            'career_pli_sum': career_pli_sum,
                            'career_inli_avg': career_inli_avg,
                            'career_inli_sum': career_inli_sum,
                            'career_gmli_avg': career_gmli_avg,
                            'career_gmli_sum': career_gmli_sum,
                            'career_exli_avg': career_exli_avg,
                            'career_exli_sum': career_exli_sum,
                            'career_clutch_avg': career_clutch_avg,
                            'career_clutch_sum': career_clutch_sum,
                            'career_siera_avg': career_siera_avg,
                            'career_siera_sum': career_siera_sum,
                            'career_pace_avg': career_pace_avg,
                            'career_pace_sum': career_pace_sum,
                            }
            for k, v in career_stats.items():
                if type(v) == np.int64:
                    v = int(v)
                elif type(v) == np.float64:
                    v = float(v)
                career_stats[k] = v
        else:
            career_stats = {'seasons_played': 0,
                            'current_age': 0,
                            'career_fb_pct': 0,
                            'career_sl_pct': 0,
                            'career_ct_pct': 0,
                            'career_cb_pct': 0,
                            'career_ch_pct': 0,
                            'career_sf_pct': 0,
                            'career_kn_pct': 0,
                            'career_xx_pct': 0,
                            'career_fbv': 0,
                            'career_slv': 0,
                            'career_ctv': 0,
                            'career_cbv': 0,
                            'career_chv': 0,
                            'career_sfv': 0,
                            'career_knv': 0,
                            'career_opp_avg': 0,
                            # 'career_wp_pct': 0,
                            'career_strike_pct': 0,
                            'career_ball_pct': 0,
                            'career_zone_pct': 0,
                            'career_swstr_pct': 0,
                            'career_pull_pct': 0,
                            'career_cent_pct': 0,
                            'career_oppo_pct': 0,
                            'career_soft_pct': 0,
                            'career_med_pct': 0,
                            'career_hard_pct': 0,
                            'career_f_strike_pct': 0,
                            'career_era': 0,
                            'career_rs': 0,
                            'career_o_swing_pct': 0,
                            'career_z_swing_pct': 0,
                            'career_o_contact_pct': 0,
                            'career_z_contact_pct': 0,
                            # ADDED V5
                            'career_war_avg': 0,
                            'career_war_sum': 0,
                            # ADDED V6
                            'career_starting_avg': 0,
                            'career_starting_sum': 0,
                            'career_start_ip_avg': 0,
                            'career_start_ip_sum': 0,
                            'career_relieving_avg': 0,
                            'career_relieving_sum': 0,
                            'career_relief_ip_avg': 0,
                            'career_relief_ip_sum': 0,
                            'career_wfb_avg': 0,
                            'career_wfb_sum': 0,
                            'career_wsl_avg': 0,
                            'career_wsl_sum': 0,
                            'career_wct_avg': 0,
                            'career_wct_sum': 0,
                            'career_wcb_avg': 0,
                            'career_wcb_sum': 0,
                            'career_wch_avg': 0,
                            'career_wch_sum': 0,
                            'career_wsf_avg': 0,
                            'career_wsf_sum': 0,
                            'career_wkn_avg': 0,
                            'career_wkn_sum': 0,
                            # REWORKED
                            'career_ip_sum': 0,
                            'career_ip_avg': 0,
                            'career_tbf_sum': 0,
                            'career_tbf_avg': 0,
                            'career_pitches_sum': 0,
                            'career_pitches_avg': 0,
                            'career_swing_pct_sum': 0,
                            'career_swing_pct_avg': 0,
                            'career_contact_pct_sum': 0,
                            'career_contact_pct_avg': 0,
                            'career_h_sum': 0,
                            'career_h_avg': 0,
                            'career_bb_sum': 0,
                            'career_bb_avg': 0,
                            'career_wp_sum': 0,
                            'career_wp_avg': 0,
                            'career_pulls_sum': 0,
                            'career_pulls_avg': 0,
                            'career_bk_sum': 0,
                            'career_bk_avg': 0,
                            'career_so_sum': 0,
                            'career_so_avg': 0,
                            'career_gb_sum': 0,
                            'career_gb_avg': 0,
                            'career_fb_sum': 0,
                            'career_fb_avg': 0,
                            'career_ld_sum': 0,
                            'career_ld_avg': 0,
                            'career_hld_avg': 0,
                            'career_hld_sum': 0,
                            'career_w_avg': 0,
                            'career_w_sum': 0,
                            'career_l_avg': 0,
                            'career_l_sum': 0,
                            'career_g_avg': 0,
                            'career_g_sum': 0,
                            'career_gs_avg': 0,
                            'career_gs_sum': 0,
                            'career_cg_avg': 0,
                            'career_cg_sum': 0,
                            'career_sho_avg': 0,
                            'career_sho_sum': 0,
                            'career_sv_avg': 0,
                            'career_sv_sum': 0,
                            'career_bs_avg': 0,
                            'career_bs_sum': 0,
                            'career_whip_avg': 0,
                            'career_whip_sum': 0,
                            'career_babip_avg': 0,
                            'career_babip_sum': 0,
                            'career_lob_pct_avg': 0,
                            'career_lob_pct_sum': 0,
                            'career_fip_avg': 0,
                            'career_fip_sum': 0,
                            'career_rar_avg': 0,
                            'career_rar_sum': 0,
                            'career_tera_avg': 0,
                            'career_tera_sum': 0,
                            'career_xfip_avg': 0,
                            'career_xfip_sum': 0,
                            'career_neg_wpa_avg': 0,
                            'career_neg_wpa_sum': 0,
                            'career_pos_wpa_avg': 0,
                            'career_pos_wpa_sum': 0,
                            'career_re24_avg': 0,
                            'career_re24_sum': 0,
                            'career_rew_avg': 0,
                            'career_rew_sum': 0,
                            'career_pli_avg': 0,
                            'career_pli_sum': 0,
                            'career_inli_avg': 0,
                            'career_inli_sum': 0,
                            'career_gmli_avg': 0,
                            'career_gmli_sum': 0,
                            'career_exli_avg': 0,
                            'career_exli_sum': 0,
                            'career_clutch_avg': 0,
                            'career_clutch_sum': 0,
                            'career_siera_avg': 0,
                            'career_siera_sum': 0,
                            'career_pace_avg': 0,
                            'career_pace_sum': 0,
                            }
        return career_stats

    def get_batter_career_stats(self, batter_id, ab_year):
        """
        Get career statistics for a given batter up until a certain year
        :param batter_id: ID of batter for which stats are requested
        :param ab_year: "current" year. Statistics for years *before* ab_year will be returned
        :return: Career statistics for a specified player up until a certain year
        """
        b_bio_info = self.statcast_id_to_bio_info[str(batter_id)]
        b_first_played = int(b_bio_info['mlb_played_first'])
        b_last_played = int(b_bio_info['mlb_played_last'])
        b_fullname = b_bio_info['fangraphs_name']

        if b_fullname == 'UNK':
            b_fname = b_bio_info['name_first']
            b_lname = b_bio_info['name_last']

            b_fname = b_fname.replace('.', '%')
            b_fname = b_fname.replace(' ', '')
            b_fname = b_fname.replace('é', 'e')
            b_lname = b_lname.replace('.', '%')
            b_lname = b_lname.replace(' ', '')
            b_lname = b_lname.replace('é', 'e')
            b_fullname = '{} {}%'.format(b_fname, b_lname)
            raw_fullname = b_fullname[:]
            b_fullname = self.troublesome_names.get(b_fullname, b_fullname)
            if b_fullname != raw_fullname:
                print('\t!! {} mapped to {} !!'.format(raw_fullname, b_fullname))

        attr_names = ['Season', 'Team', 'Age', 'G', 'AB', 'PA', 'H', 'B1', 'B2', 'B3', 'HR', 'R', 'RBI', 'BB', 'IBB',
                      'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'GB', 'FB', 'LD', 'IFFB', 'Pitches', 'Balls',
                      'Strikes', 'IFH', 'BU', 'BUH', 'PH', 'FB_pct_pitch', 'FBv', 'SL_pct', 'SLv', 'CT_pct', 'CTv',
                      'CB_pct', 'CBv', 'CH_pct', 'CHv', 'SF_pct', 'SFv', 'KN_pct', 'KNv', 'XX_pct', 'O_Swing_pct',
                      'Z_Swing_pct', 'Swing_pct', 'O_Contact_pct', 'Z_Contact_pct', 'Contact_pct', 'Zone_pct',
                      'F_Strike_pct', 'SwStr_pct', 'Pull_pct', 'Cent_pct', 'Oppo_pct', 'Soft_pct', 'Med_pct',
                      'Hard_pct',
                      # new for v5
                      'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'wOBA', 'wRAA', 'wRC', 'WAR', 'RAR', 'WPA', '_WPA',
                      'pos_WPA', 'RE24', 'REW', 'pLI', 'phLI', 'Clutch', 'BsR',
                      # new for v6
                      'Bat', 'Fld', 'Rep', 'Pos', 'Spd', 'wFB', 'wSL', 'wCT', 'wCB', 'wCH', 'wSF', 'wKN',
                      'Pace', 'Def', 'wSB', 'UBR', 'Off', 'Lg', 'wGDP'
                      ]
        batter_career_data_query = """SELECT Season, Team, Age, G, AB, PA, G, B1, B2, B3, HR, R, RBI, BB, IBB, SO, HBP,
                                             SF, SH, GDP, SB, CS, GB, FB, LD, IFFB, Pitches, Balls, Strikes, IFH, BU,
                                             BUH, PH, FB_pct_pitch, FBv, SL_pct, SLv, CT_pct, CTv, CB_pct, CBv, CH_pct, 
                                             CHv, SF_pct, SFv, KN_pct, KNv, XX_pct, O_Swing_pct, Z_Swing_pct, 
                                             Swing_pct, O_Contact_pct, Z_Contact_pct, Contact_pct, Zone_pct, 
                                             F_Strike_pct, SwStr_pct, Pull_pct, Cent_pct, Oppo_pct, Soft_pct,
                                             Med_pct, Hard_pct, OBP, SLG, OPS, ISO, BABIP, wOBA, wRAA, wRC, WAR, RAR,
                                             WPA, _WPA, pos_WPA, RE24, REW, pLI, phLI, Clutch, BsR,
                                             Bat, Fld, Rep, Pos, Spd, wFB, wSL, wCT, wCB, wCH, wSF, wKN, Pace,
                                             Def, wSB, UBR, Off, Lg, wGDP
                                       FROM batting_by_season
                                       WHERE LOWER(Name) LIKE ?
                                       AND Season >= ?
                                       AND Season <= ?
                                       AND Season < ?"""
        query_args = (b_fullname, b_first_played, b_last_played, ab_year)
        raw_batter_career_data = self.query_db(batter_career_data_query, query_args)
        batter_df = pd.DataFrame(raw_batter_career_data, columns=attr_names).fillna(0)

        if batter_df.shape[0] > 0:
            seasons_played = batter_df.shape[0]
            current_age = int(batter_df['Age'].max()) + 1

            career_o_swing_pct = batter_df['O_Swing_pct'].mean()
            career_z_swing_pct = batter_df['Z_Swing_pct'].mean()
            career_o_contact_pct = batter_df['O_Contact_pct'].mean()
            career_z_contact_pct = batter_df['Z_Contact_pct'].mean()
            career_swing_pct_sum = batter_df['Swing_pct'].sum()
            career_swing_pct_avg = batter_df['Swing_pct'].mean()
            career_contact_pct_sum = batter_df['Contact_pct'].sum()
            career_contact_pct_avg = batter_df['Contact_pct'].mean()
            # career_swing_pct = batter_df['Swing_pct'].mean()
            # career_contact_pct = batter_df['Contact_pct'].mean()

            career_games_avg = batter_df['G'].mean()
            career_games_sum = batter_df['G'].sum()
            career_ab_avg = batter_df['AB'].mean()
            career_ab_sum = batter_df['AB'].sum()
            career_pa_avg = batter_df['PA'].mean()
            career_pa_sum = batter_df['PA'].sum()
            career_h_avg = batter_df['H'].mean()
            career_h_sum = batter_df['H'].sum()
            career_b1_avg = batter_df['B1'].mean()
            career_b1_sum = batter_df['B1'].sum()
            career_b2_avg = batter_df['B2'].mean()
            career_b2_sum = batter_df['B2'].sum()
            career_b3_avg = batter_df['B3'].mean()
            career_b3_sum = batter_df['B3'].sum()
            career_hr_avg = batter_df['HR'].mean()
            career_hr_sum = batter_df['HR'].sum()
            career_r_avg = batter_df['R'].mean()
            career_r_sum = batter_df['R'].sum()
            career_rbi_avg = batter_df['RBI'].mean()
            career_rbi_sum = batter_df['RBI'].sum()
            career_bb_avg = batter_df['BB'].mean()
            career_bb_sum = batter_df['BB'].sum()
            career_ibb_avg = batter_df['IBB'].mean()
            career_ibb_sum = batter_df['IBB'].sum()
            career_so_avg = batter_df['SO'].mean()
            career_so_sum = batter_df['SO'].sum()
            career_hbp_avg = batter_df['HBP'].mean()
            career_hbp_sum = batter_df['HBP'].sum()
            career_sf_avg = batter_df['SF'].mean()
            career_sf_sum = batter_df['SF'].sum()
            career_sh_avg = batter_df['SH'].mean()
            career_sh_sum = batter_df['SH'].sum()
            career_gdp_avg = batter_df['GDP'].mean()
            career_gdp_sum = batter_df['GDP'].sum()
            career_iffb_avg = batter_df['IFFB'].mean()
            career_iffb_sum = batter_df['IFFB'].sum()
            career_sb_avg = batter_df['SB'].mean()
            career_sb_sum = batter_df['SB'].sum()
            career_cs_avg = batter_df['CS'].mean()
            career_cs_sum = batter_df['CS'].sum()
            career_ifh_avg = batter_df['IFH'].mean()
            career_ifh_sum = batter_df['IFH'].sum()
            career_bu_avg = batter_df['BU'].mean()
            career_bu_sum = batter_df['BU'].sum()
            career_buh_avg = batter_df['BUH'].mean()
            career_buh_sum = batter_df['BUH'].sum()
            career_ph_avg = batter_df['PH'].mean()
            career_ph_sum = batter_df['PH'].sum()

            career_zone_pct = batter_df['Zone_pct'].mean()
            career_f_strike_pct = batter_df['F_Strike_pct'].mean()
            career_pull_pct = batter_df['Pull_pct'].mean()
            career_cent_pct = batter_df['Cent_pct'].mean()
            career_oppo_pct = batter_df['Oppo_pct'].mean()
            career_soft_pct = batter_df['Soft_pct'].mean()
            career_med_pct = batter_df['Med_pct'].mean()
            career_hard_pct = batter_df['Hard_pct'].mean()
            career_swstr_pct = batter_df['SwStr_pct'].mean()

            # Pitch distribution
            career_fb_pct = batter_df['FB_pct_pitch'].mean()
            career_sl_pct = batter_df['SL_pct'].mean()
            career_ct_pct = batter_df['CT_pct'].mean()
            career_cb_pct = batter_df['CB_pct'].mean()
            career_ch_pct = batter_df['CH_pct'].mean()
            career_sf_pct = batter_df['SF_pct'].mean()
            career_kn_pct = batter_df['KN_pct'].mean()
            career_xx_pct = batter_df['XX_pct'].mean()

            # Pitch velocities
            career_fbv = batter_df['FBv'].mean()
            career_slv = batter_df['SLv'].mean()
            career_ctv = batter_df['CTv'].mean()
            career_cbv = batter_df['CBv'].mean()
            career_chv = batter_df['CHv'].mean()
            career_sfv = batter_df['SFv'].mean()
            career_knv = batter_df['KNv'].mean()

            # batter_df['Groundball_pct'] = batter_df['GB'] / (batter_df['GB'] + batter_df['FB'] + batter_df['LD'])
            # batter_df['Flyball_pct'] = batter_df['FB'] / (batter_df['GB'] + batter_df['FB'] + batter_df['LD'])
            # batter_df['Linedrive_pct'] = batter_df['LD'] / (batter_df['GB'] + batter_df['FB'] + batter_df['LD'])
            batter_df['Ball_pct'] = batter_df['Balls'] / batter_df['Pitches']
            batter_df['Strike_pct'] = batter_df['Strikes'] / batter_df['Pitches']
            batter_df['Batting_avg'] = batter_df['H'] / batter_df['AB']

            career_gb_sum = batter_df['GB'].sum()
            career_gb_avg = batter_df['GB'].mean()
            career_fb_sum = batter_df['FB'].sum()
            career_fb_avg = batter_df['FB'].mean()
            career_ld_sum = batter_df['LD'].sum()
            career_ld_avg = batter_df['LD'].mean()

            # career_groundball_pct = batter_df['Groundball_pct'].mean()
            # career_flyball_pct = batter_df['Flyball_pct'].mean()
            # career_linedrive_pct = batter_df['Linedrive_pct'].mean()
            career_ball_pct = batter_df['Ball_pct'].mean()
            career_strike_pct = batter_df['Strike_pct'].mean()
            career_batting_avg = batter_df['Batting_avg'].mean()

            # v5
            career_war_avg = batter_df['WAR'].mean()
            career_war_sum = batter_df['WAR'].sum()
            career_obp_avg = batter_df['OBP'].mean()
            career_obp_sum = batter_df['OBP'].sum()
            career_slg_avg = batter_df['SLG'].mean()
            career_slg_sum = batter_df['SLG'].sum()
            career_ops_avg = batter_df['OPS'].mean()
            career_ops_sum = batter_df['OPS'].sum()
            career_iso_avg = batter_df['ISO'].mean()
            career_iso_sum = batter_df['ISO'].sum()
            career_babip_avg = batter_df['BABIP'].mean()
            career_babip_sum = batter_df['BABIP'].sum()
            career_woba_avg = batter_df['wOBA'].mean()
            career_woba_sum = batter_df['wOBA'].sum()
            career_wraa_avg = batter_df['wRAA'].mean()
            career_wraa_sum = batter_df['wRAA'].sum()
            career_wrc_avg = batter_df['wRC'].mean()
            career_wrc_sum = batter_df['wRC'].sum()

            career_rar_sum = batter_df['RAR'].sum()
            career_rar_avg = batter_df['RAR'].mean()
            career_wpa_sum = batter_df['WPA'].sum()
            career_wpa_avg = batter_df['WPA'].mean()
            career_neg_wpa_sum = batter_df['_WPA'].sum()
            career_neg_wpa_avg = batter_df['_WPA'].mean()
            career_pos_wpa_sum = batter_df['pos_WPA'].sum()
            career_pos_wpa_avg = batter_df['pos_WPA'].mean()
            career_re24_sum = batter_df['RE24'].sum()
            career_re24_avg = batter_df['RE24'].mean()
            career_rew_sum = batter_df['REW'].sum()
            career_rew_avg = batter_df['REW'].mean()
            career_pli_sum = batter_df['pLI'].sum()
            career_pli_avg = batter_df['pLI'].mean()
            career_phli_sum = batter_df['phLI'].sum()
            career_phli_avg = batter_df['phLI'].mean()
            career_clutch_sum = batter_df['Clutch'].sum()
            career_clutch_avg = batter_df['Clutch'].mean()
            career_bsr_sum = batter_df['BsR'].sum()
            career_bsr_avg = batter_df['BsR'].mean()

            # v6
            career_bat_avg = batter_df['Bat'].mean()
            career_bat_sum = batter_df['Bat'].sum()
            career_fld_avg = batter_df['Fld'].mean()
            career_fld_sum = batter_df['Fld'].sum()
            career_rep_avg = batter_df['Rep'].mean()
            career_rep_sum = batter_df['Rep'].sum()
            career_pos_avg = batter_df['Pos'].mean()
            career_pos_sum = batter_df['Pos'].sum()
            career_spd_avg = batter_df['Spd'].mean()
            career_spd_sum = batter_df['Spd'].sum()
            career_wfb_avg = batter_df['wFB'].mean()
            career_wfb_sum = batter_df['wFB'].sum()
            career_wsl_avg = batter_df['wSL'].mean()
            career_wsl_sum = batter_df['wSL'].sum()
            career_wct_avg = batter_df['wCT'].mean()
            career_wct_sum = batter_df['wCT'].sum()
            career_wcb_avg = batter_df['wCB'].mean()
            career_wcb_sum = batter_df['wCB'].sum()
            career_wch_avg = batter_df['wCH'].mean()
            career_wch_sum = batter_df['wCH'].sum()
            career_wsf_avg = batter_df['wSF'].mean()
            career_wsf_sum = batter_df['wSF'].sum()
            career_wkn_avg = batter_df['wKN'].mean()
            career_wkn_sum = batter_df['wKN'].sum()
            career_pace_avg = batter_df['Pace'].mean()
            career_pace_sum = batter_df['Pace'].sum()
            career_def_avg = batter_df['Def'].mean()
            career_def_sum = batter_df['Def'].sum()
            career_wsb_avg = batter_df['wSB'].mean()
            career_wsb_sum = batter_df['wSB'].sum()
            career_ubr_avg = batter_df['UBR'].mean()
            career_ubr_sum = batter_df['UBR'].sum()
            career_off_avg = batter_df['Off'].mean()
            career_off_sum = batter_df['Off'].sum()
            career_lg_avg = batter_df['Lg'].mean()
            career_lg_sum = batter_df['Lg'].sum()
            career_wgdp_avg = batter_df['wGDP'].mean()
            career_wgdp_sum = batter_df['wGDP'].sum()

            career_stats = {'season_played': seasons_played,
                            'current_age': current_age,
                            'career_fb_pct_pitch': career_fb_pct,
                            'career_sl_pct': career_sl_pct,
                            'career_ct_pct': career_ct_pct,
                            'career_cb_pct': career_cb_pct,
                            'career_ch_pct': career_ch_pct,
                            'career_sf_pct_pitch': career_sf_pct,
                            'career_kn_pct': career_kn_pct,
                            'career_xx_pct': career_xx_pct,
                            'career_fbv': career_fbv,
                            'career_slv': career_slv,
                            'career_ctv': career_ctv,
                            'career_cbv': career_cbv,
                            'career_chv': career_chv,
                            'career_sfv': career_sfv,
                            'career_knv': career_knv,
                            'career_strike_pct': career_strike_pct,
                            'career_ball_pct': career_ball_pct,
                            'career_zone_pct': career_zone_pct,
                            'career_swstr_pct': career_swstr_pct,
                            'career_pull_pct': career_pull_pct,
                            'career_cent_pct': career_cent_pct,
                            'career_oppo_pct': career_oppo_pct,
                            'career_soft_pct': career_soft_pct,
                            'career_med_pct': career_med_pct,
                            'career_hard_pct': career_hard_pct,
                            'career_f_strike_pct': career_f_strike_pct,
                            'career_avg': career_batting_avg,
                            'career_o_swing_pct': career_o_swing_pct,
                            'career_z_swing_pct': career_z_swing_pct,
                            'career_o_contact_pct': career_o_contact_pct,
                            'career_z_contact_pct': career_z_contact_pct,
                            # new v5
                            'career_war_avg': career_war_avg,
                            'career_war_sum': career_war_sum,
                            # new v6
                            'career_bat_avg': career_bat_avg,
                            'career_bat_sum': career_bat_sum,
                            'career_fld_avg': career_fld_avg,
                            'career_fld_sum': career_fld_sum,
                            'career_rep_avg': career_rep_avg,
                            'career_rep_sum': career_rep_sum,
                            'career_pos_avg': career_pos_avg,
                            'career_pos_sum': career_pos_sum,
                            'career_spd_avg': career_spd_avg,
                            'career_spd_sum': career_spd_sum,
                            'career_wfb_avg': career_wfb_avg,
                            'career_wfb_sum': career_wfb_sum,
                            'career_wsl_avg': career_wsl_avg,
                            'career_wsl_sum': career_wsl_sum,
                            'career_wct_avg': career_wct_avg,
                            'career_wct_sum': career_wct_sum,
                            'career_wcb_avg': career_wcb_avg,
                            'career_wcb_sum': career_wcb_sum,
                            'career_wch_avg': career_wch_avg,
                            'career_wch_sum': career_wch_sum,
                            'career_wsf_avg': career_wsf_avg,
                            'career_wsf_sum': career_wsf_sum,
                            'career_wkn_avg': career_wkn_avg,
                            'career_wkn_sum': career_wkn_sum,
                            'career_pace_avg': career_pace_avg,
                            'career_pace_sum': career_pace_sum,
                            'career_def_avg': career_def_avg,
                            'career_def_sum': career_def_sum,
                            'career_wsb_avg': career_wsb_avg,
                            'career_wsb_sum': career_wsb_sum,
                            'career_ubr_avg': career_ubr_avg,
                            'career_ubr_sum': career_ubr_sum,
                            'career_off_avg': career_off_avg,
                            'career_off_sum': career_off_sum,
                            'career_lg_avg': career_lg_avg,
                            'career_lg_sum': career_lg_sum,
                            'career_wgdp_avg': career_wgdp_avg,
                            'career_wgdp_sum': career_wgdp_sum,
                            # REWORKED
                            'career_swing_pct_sum': career_swing_pct_sum,
                            'career_swing_pct_avg': career_swing_pct_avg,
                            'career_contact_pct_sum': career_contact_pct_sum,
                            'career_contact_pct_avg': career_contact_pct_avg,
                            'career_games_avg': career_games_avg,
                            'career_games_sum': career_games_sum,
                            'career_ab_avg': career_ab_avg,
                            'career_ab_sum': career_ab_sum,
                            'career_pa_avg': career_pa_avg,
                            'career_pa_sum': career_pa_sum,
                            'career_h_avg': career_h_avg,
                            'career_h_sum': career_h_sum,
                            'career_b1_avg': career_b1_avg,
                            'career_b1_sum': career_b1_sum,
                            'career_b2_avg': career_b2_avg,
                            'career_b2_sum': career_b2_sum,
                            'career_b3_avg': career_b3_avg,
                            'career_b3_sum': career_b3_sum,
                            'career_hr_avg': career_hr_avg,
                            'career_hr_sum': career_hr_sum,
                            'career_r_avg': career_r_avg,
                            'career_r_sum': career_r_sum,
                            'career_rbi_avg': career_rbi_avg,
                            'career_rbi_sum': career_rbi_sum,
                            'career_bb_avg': career_bb_avg,
                            'career_bb_sum': career_bb_sum,
                            'career_ibb_avg': career_ibb_avg,
                            'career_ibb_sum': career_ibb_sum,
                            'career_so_avg': career_so_avg,
                            'career_so_sum': career_so_sum,
                            'career_hbp_avg': career_hbp_avg,
                            'career_hbp_sum': career_hbp_sum,
                            'career_sf_avg': career_sf_avg,
                            'career_sf_sum': career_sf_sum,
                            'career_sh_avg': career_sh_avg,
                            'career_sh_sum': career_sh_sum,
                            'career_gdp_avg': career_gdp_avg,
                            'career_gdp_sum': career_gdp_sum,
                            'career_iffb_avg': career_iffb_avg,
                            'career_iffb_sum': career_iffb_sum,
                            'career_sb_avg': career_sb_avg,
                            'career_sb_sum': career_sb_sum,
                            'career_cs_avg': career_cs_avg,
                            'career_cs_sum': career_cs_sum,
                            'career_ifh_avg': career_ifh_avg,
                            'career_ifh_sum': career_ifh_sum,
                            'career_bu_avg': career_bu_avg,
                            'career_bu_sum': career_bu_sum,
                            'career_buh_avg': career_buh_avg,
                            'career_buh_sum': career_buh_sum,
                            'career_ph_avg': career_ph_avg,
                            'career_ph_sum': career_ph_sum,
                            'career_gb_sum': career_gb_sum,
                            'career_gb_avg': career_gb_avg,
                            'career_fb_sum': career_fb_sum,
                            'career_fb_avg': career_fb_avg,
                            'career_ld_sum': career_ld_sum,
                            'career_ld_avg': career_ld_avg,
                            'career_obp_avg': career_obp_avg,
                            'career_obp_sum': career_obp_sum,
                            'career_slg_avg': career_slg_avg,
                            'career_slg_sum': career_slg_sum,
                            'career_ops_avg': career_ops_avg,
                            'career_ops_sum': career_ops_sum,
                            'career_iso_avg': career_iso_avg,
                            'career_iso_sum': career_iso_sum,
                            'career_babip_avg': career_babip_avg,
                            'career_babip_sum': career_babip_sum,
                            'career_woba_avg': career_woba_avg,
                            'career_woba_sum': career_woba_sum,
                            'career_wraa_avg': career_wraa_avg,
                            'career_wraa_sum': career_wraa_sum,
                            'career_wrc_avg': career_wrc_avg,
                            'career_wrc_sum': career_wrc_sum,
                            'career_rar_sum': career_rar_sum,
                            'career_rar_avg': career_rar_avg,
                            'career_wpa_sum': career_wpa_sum,
                            'career_wpa_avg': career_wpa_avg,
                            'career_neg_wpa_sum': career_neg_wpa_sum,
                            'career_neg_wpa_avg': career_neg_wpa_avg,
                            'career_pos_wpa_sum': career_pos_wpa_sum,
                            'career_pos_wpa_avg': career_pos_wpa_avg,
                            'career_re24_sum': career_re24_sum,
                            'career_re24_avg': career_re24_avg,
                            'career_rew_sum': career_rew_sum,
                            'career_rew_avg': career_rew_avg,
                            'career_pli_sum': career_pli_sum,
                            'career_pli_avg': career_pli_avg,
                            'career_phli_sum': career_phli_sum,
                            'career_phli_avg': career_phli_avg,
                            'career_clutch_sum': career_clutch_sum,
                            'career_clutch_avg': career_clutch_avg,
                            'career_bsr_sum': career_bsr_sum,
                            'career_bsr_avg': career_bsr_avg,
                            }
            for k, v in career_stats.items():
                if type(v) == np.int64:
                    v = int(v)
                elif type(v) == np.float64:
                    v = float(v)
                career_stats[k] = v
        else:
            career_stats = {'season_played': 0,
                            'current_age': 0,
                            'career_fb_pct_pitch': 0,
                            'career_sl_pct': 0,
                            'career_ct_pct': 0,
                            'career_cb_pct': 0,
                            'career_ch_pct': 0,
                            'career_sf_pct_pitch': 0,
                            'career_kn_pct': 0,
                            'career_xx_pct': 0,
                            'career_fbv': 0,
                            'career_slv': 0,
                            'career_ctv': 0,
                            'career_cbv': 0,
                            'career_chv': 0,
                            'career_sfv': 0,
                            'career_knv': 0,
                            'career_strike_pct': 0,
                            'career_ball_pct': 0,
                            'career_zone_pct': 0,
                            'career_swstr_pct': 0,
                            'career_pull_pct': 0,
                            'career_cent_pct': 0,
                            'career_oppo_pct': 0,
                            'career_soft_pct': 0,
                            'career_med_pct': 0,
                            'career_hard_pct': 0,
                            'career_f_strike_pct': 0,
                            'career_avg': 0,
                            'career_o_swing_pct': 0,
                            'career_z_swing_pct': 0,
                            'career_o_contact_pct': 0,
                            'career_z_contact_pct': 0,
                            # new v5
                            'career_war_avg': 0,
                            'career_war_sum': 0,
                            # new v6
                            'career_bat_avg': 0,
                            'career_bat_sum': 0,
                            'career_fld_avg': 0,
                            'career_fld_sum': 0,
                            'career_rep_avg': 0,
                            'career_rep_sum': 0,
                            'career_pos_avg': 0,
                            'career_pos_sum': 0,
                            'career_spd_avg': 0,
                            'career_spd_sum': 0,
                            'career_wfb_avg': 0,
                            'career_wfb_sum': 0,
                            'career_wsl_avg': 0,
                            'career_wsl_sum': 0,
                            'career_wct_avg': 0,
                            'career_wct_sum': 0,
                            'career_wcb_avg': 0,
                            'career_wcb_sum': 0,
                            'career_wch_avg': 0,
                            'career_wch_sum': 0,
                            'career_wsf_avg': 0,
                            'career_wsf_sum': 0,
                            'career_wkn_avg': 0,
                            'career_wkn_sum': 0,
                            'career_pace_avg': 0,
                            'career_pace_sum': 0,
                            'career_def_avg': 0,
                            'career_def_sum': 0,
                            'career_wsb_avg': 0,
                            'career_wsb_sum': 0,
                            'career_ubr_avg': 0,
                            'career_ubr_sum': 0,
                            'career_off_avg': 0,
                            'career_off_sum': 0,
                            'career_lg_avg': 0,
                            'career_lg_sum': 0,
                            'career_wgdp_avg': 0,
                            'career_wgdp_sum': 0,
                            # REWORKED
                            'career_swing_pct_sum': 0,
                            'career_swing_pct_avg': 0,
                            'career_contact_pct_sum': 0,
                            'career_contact_pct_avg': 0,
                            'career_games_avg': 0,
                            'career_games_sum': 0,
                            'career_ab_avg': 0,
                            'career_ab_sum': 0,
                            'career_pa_avg': 0,
                            'career_pa_sum': 0,
                            'career_h_avg': 0,
                            'career_h_sum': 0,
                            'career_b1_avg': 0,
                            'career_b1_sum': 0,
                            'career_b2_avg': 0,
                            'career_b2_sum': 0,
                            'career_b3_avg': 0,
                            'career_b3_sum': 0,
                            'career_hr_avg': 0,
                            'career_hr_sum': 0,
                            'career_r_avg': 0,
                            'career_r_sum': 0,
                            'career_rbi_avg': 0,
                            'career_rbi_sum': 0,
                            'career_bb_avg': 0,
                            'career_bb_sum': 0,
                            'career_ibb_avg': 0,
                            'career_ibb_sum': 0,
                            'career_so_avg': 0,
                            'career_so_sum': 0,
                            'career_hbp_avg': 0,
                            'career_hbp_sum': 0,
                            'career_sf_avg': 0,
                            'career_sf_sum': 0,
                            'career_sh_avg': 0,
                            'career_sh_sum': 0,
                            'career_gdp_avg': 0,
                            'career_gdp_sum': 0,
                            'career_iffb_avg': 0,
                            'career_iffb_sum': 0,
                            'career_sb_avg': 0,
                            'career_sb_sum': 0,
                            'career_cs_avg': 0,
                            'career_cs_sum': 0,
                            'career_ifh_avg': 0,
                            'career_ifh_sum': 0,
                            'career_bu_avg': 0,
                            'career_bu_sum': 0,
                            'career_buh_avg': 0,
                            'career_buh_sum': 0,
                            'career_ph_avg': 0,
                            'career_ph_sum': 0,
                            'career_gb_sum': 0,
                            'career_gb_avg': 0,
                            'career_fb_sum': 0,
                            'career_fb_avg': 0,
                            'career_ld_sum': 0,
                            'career_ld_avg': 0,
                            'career_obp_avg': 0,
                            'career_obp_sum': 0,
                            'career_slg_avg': 0,
                            'career_slg_sum': 0,
                            'career_ops_avg': 0,
                            'career_ops_sum': 0,
                            'career_iso_avg': 0,
                            'career_iso_sum': 0,
                            'career_babip_avg': 0,
                            'career_babip_sum': 0,
                            'career_woba_avg': 0,
                            'career_woba_sum': 0,
                            'career_wraa_avg': 0,
                            'career_wraa_sum': 0,
                            'career_wrc_avg': 0,
                            'career_wrc_sum': 0,
                            'career_rar_sum': 0,
                            'career_rar_avg': 0,
                            'career_wpa_sum': 0,
                            'career_wpa_avg': 0,
                            'career_neg_wpa_sum': 0,
                            'career_neg_wpa_avg': 0,
                            'career_pos_wpa_sum': 0,
                            'career_pos_wpa_avg': 0,
                            'career_re24_sum': 0,
                            'career_re24_avg': 0,
                            'career_rew_sum': 0,
                            'career_rew_avg': 0,
                            'career_pli_sum': 0,
                            'career_pli_avg': 0,
                            'career_phli_sum': 0,
                            'career_phli_avg': 0,
                            'career_clutch_sum': 0,
                            'career_clutch_avg': 0,
                            'career_bsr_sum': 0,
                            'career_bsr_avg': 0,
                            }
        return career_stats

    def get_player_season_stats(self, player_id, game_pk, year, player_type, game_date=None, ab_number=None,
                                last_n_days=365):
        """
        Get player statistics *within* season
        :param player_id: ID of player being queried
        :param game_pk: ID of game that stats are being requested for. No stats from this game or subsequent games will be returned
        :param year: year stats are being queried for
        :param player_type: type of player being queried (pitcher or batter)
        :param game_date: date of game that stats are being queried for. No stats after this date will be returned
        :param ab_number: at-bat number for which stats are being queried for. No stats after this at-bat will be returned
        :param last_n_days: how many days before given game/at-bat to include in query
        :return: stats describing player in season up to given point in time
        """
        if player_type == 'pitcher':
            if game_date is None or ab_number is None:
                query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                                  home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                                  at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                            from statcast 
                            where pitcher = ?
                            and game_year = ?
                            and game_pk < ?"""
            elif last_n_days == 0:
                query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                                  home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                                  at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                            from statcast 
                            where pitcher = ?
                            and game_pk = ? and at_bat_number < ?"""
            else:
                query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                                  home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                                  at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                            from statcast 
                            where pitcher = ?
                            and game_year = ?
                            and (days_since_2000 between julianday(?) - julianday(?) - ? and julianday(?) - julianday(?) - 1 
                            or (game_pk = ? and at_bat_number < ?))"""
        else:
            if game_date is None or ab_number is None:
                query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                                  home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                                  at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                            from statcast 
                            where batter = ?
                            and game_year = ?
                            and game_pk < ?"""
            elif last_n_days == 0:
                query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                                  home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                                  at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                            from statcast 
                            where batter = ?
                            and game_pk = ? and at_bat_number < ?"""
            else:
                query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                                  home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                                  at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                            from statcast 
                            where batter = ?
                            and game_year = ?
                            and (days_since_2000 between julianday(?) - julianday(?) - ? and julianday(?) - julianday(?) - 1 
                            or (game_pk = ? and at_bat_number < ?))"""

        if game_date is None or ab_number is None:
            query_args = (player_id, year, game_pk)
        elif last_n_days == 0:
            query_args = (player_id, game_pk, ab_number)
        else:
            query_args = (
                player_id, year, game_date, '2000-01-01', last_n_days, game_date, '2000-01-01', game_pk, ab_number)
        player_season_data = self.query_db(query, query_args)

        attr_names = ['pitch_type', 'game_date', 'release_speed', 'batter', 'pitcher', 'events', 'game_type', 'stand',
                      'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes',
                      'game_year', 'at_bat_number', 'sz_top', 'sz_bot', 'game_pk', 'pitch_number',
                      'plate_x', 'plate_z', 'description', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                      'hit_distance_sc', 'launch_speed', 'launch_angle', 'release_spin_rate']
        player_df = pd.DataFrame(player_season_data, columns=attr_names).fillna(0)
        player_df['plate_x_id'] = player_df['plate_x'].apply(lambda x: decode_xpos(x))
        player_df['plate_z_id'] = player_df['plate_z'].apply(lambda z: decode_zpos(z))

        desc_to_swing_status = {"ball": 0, "called_strike": 0, "blocked_ball": 0, "intent_ball": 0, "hit_by_pitch": 0,
                                "pitchout": 0, "unknown_strike": 0,
                                "foul": 1, "hit_into_play": 1, "swinging_strike": 1, "hit_into_play_no_out": 1,
                                "hit_into_play_score": 1, "foul_tip": 1, "swinging_strike_blocked": 1, "foul_bunt": 1,
                                "missed_bunt": 1, "swinging_pitchout": 1, "pitchout_hit_into_play": 1,
                                "foul_pitchout": 1, "pitchout_hit_into_play_score": 1,
                                "pitchout_hit_into_play_no_out": 1, "bunt_foul_tip": 1}

        n_xpos_ids = 8
        n_zpos_ids = 9
        n_coord_ids = 10

        if player_df.shape[0] > 0:
            x_pos_swing_res = [0 for _ in range(n_xpos_ids)]
            z_pos_swing_res = [0 for _ in range(n_zpos_ids)]
            x_pos_hit_res = [0 for _ in range(n_xpos_ids)]
            z_pos_hit_res = [0 for _ in range(n_zpos_ids)]
            x_pos_pitches = [0 for _ in range(n_xpos_ids)]
            z_pos_pitches = [0 for _ in range(n_zpos_ids)]
            n_pitches = 0

            plate_x_ids = player_df['plate_x_id'].tolist()
            plate_z_ids = player_df['plate_z_id'].tolist()
            descriptions = player_df['description'].tolist()
            types = player_df['type'].tolist()

            for x_id, z_id, desc, call_type in zip(plate_x_ids, plate_z_ids, descriptions, types):
                swing_status = desc_to_swing_status[desc]
                hit_status = 1 if call_type.strip() == 'X' else 0
                x_pos_swing_res[x_id] += swing_status
                z_pos_swing_res[z_id] += swing_status
                x_pos_hit_res[x_id] += hit_status
                z_pos_hit_res[z_id] += hit_status
                x_pos_pitches[x_id] += 1
                z_pos_pitches[z_id] += 1
                n_pitches += 1

            x_pos_swing_pcts = [x / n_pitches for x in x_pos_swing_res]
            z_pos_swing_pcts = [z / n_pitches for z in z_pos_swing_res]
            x_pos_hit_pcts = [x / n_pitches for x in x_pos_hit_res]
            z_pos_hit_pcts = [z / n_pitches for z in z_pos_hit_res]
            x_pos_pitch_pcts = [x / n_pitches for x in x_pos_pitches]
            z_pos_pitch_pcts = [z / n_pitches for z in z_pos_pitches]

            hit_xpos_res = [0 for _ in range(n_coord_ids)]
            hit_ypos_res = [0 for _ in range(n_coord_ids)]
            hit_df = player_df[(player_df['hc_x'] != 0) & (player_df['hc_y'] != 0)][['hc_x', 'hc_y']]
            hit_df['x_id'] = hit_df['hc_x'].apply(lambda x: decode_hit_coord(x))
            hit_df['y_id'] = hit_df['hc_y'].apply(lambda y: decode_hit_coord(y))
            hit_x_ids = hit_df['x_id'].tolist()
            hit_y_ids = hit_df['y_id'].tolist()
            n_hits = len(hit_x_ids)

            for x_id, y_id in zip(hit_x_ids, hit_y_ids):
                hit_xpos_res[x_id] += 1
                hit_ypos_res[y_id] += 1
            hit_xpos_res = [x / n_hits if n_hits > 0 else 0.0 for x in hit_xpos_res]
            hit_ypos_res = [y / n_hits if n_hits > 0 else 0.0 for y in hit_ypos_res]

            thrown_pitches = player_df['pitch_type'].tolist()
            pitch_types, pitch_counts = np.unique(thrown_pitches, return_counts=True)
            pitch_dict = dict(zip(pitch_types, pitch_counts))
            fc_pct = pitch_dict.get('FC', 0) / n_pitches
            ff_pct = pitch_dict.get('FF', 0) / n_pitches
            sl_pct = pitch_dict.get('SL', 0) / n_pitches
            ch_pct = pitch_dict.get('CH', 0) / n_pitches
            cu_pct = pitch_dict.get('CU', 0) / n_pitches
            si_pct = pitch_dict.get('SI', 0) / n_pitches
            fs_pct = pitch_dict.get('FS', 0) / n_pitches
            ft_pct = pitch_dict.get('FT', 0) / n_pitches
            kc_pct = pitch_dict.get('KC', 0) / n_pitches
            po_pct = pitch_dict.get('PO', 0) / n_pitches
            in_pct = pitch_dict.get('IN', 0) / n_pitches
            sc_pct = pitch_dict.get('SC', 0) / n_pitches
            fa_pct = pitch_dict.get('FA', 0) / n_pitches
            ep_pct = pitch_dict.get('EP', 0) / n_pitches
            kn_pct = pitch_dict.get('KN', 0) / n_pitches
            fo_pct = pitch_dict.get('FO', 0) / n_pitches
            un_pct = pitch_dict.get('UN', 0) / n_pitches

            fc_cnt = pitch_dict.get('FC', 0)
            ff_cnt = pitch_dict.get('FF', 0)
            sl_cnt = pitch_dict.get('SL', 0)
            ch_cnt = pitch_dict.get('CH', 0)
            cu_cnt = pitch_dict.get('CU', 0)
            si_cnt = pitch_dict.get('SI', 0)
            fs_cnt = pitch_dict.get('FS', 0)
            ft_cnt = pitch_dict.get('FT', 0)
            kc_cnt = pitch_dict.get('KC', 0)
            po_cnt = pitch_dict.get('PO', 0)
            in_cnt = pitch_dict.get('IN', 0)
            sc_cnt = pitch_dict.get('SC', 0)
            fa_cnt = pitch_dict.get('FA', 0)
            ep_cnt = pitch_dict.get('EP', 0)
            kn_cnt = pitch_dict.get('KN', 0)
            fo_cnt = pitch_dict.get('FO', 0)
            un_cnt = pitch_dict.get('UN', 0)

            sz_bots = [szb for szb in player_df['sz_bot'].tolist() if not szb == 0]
            avg_sz_bot = sum(sz_bots) / len(sz_bots) if not len(sz_bots) == 0 else 0

            sz_tops = [szt for szt in player_df['sz_top'].tolist() if not szt == 0]
            avg_sz_top = sum(sz_tops) / len(sz_tops) if not len(sz_tops) == 0 else 0

            pa_ids = ['{}-{}'.format(game_pk, ab_no) for game_pk, ab_no in zip(player_df['game_pk'].tolist(),
                                                                               player_df['at_bat_number'].tolist())]
            u_pa_ids = list(np.unique(pa_ids))
            n_pa = len(u_pa_ids)
            ab_events = player_df['events'].tolist()
            event_types, event_counts = np.unique(ab_events, return_counts=True)
            event_dict = dict(zip(event_types, event_counts))

            single_pct = event_dict.get('single', 0) / n_pa
            double_pct = event_dict.get('double', 0) / n_pa
            triple_pct = event_dict.get('triple', 0) / n_pa
            hr_pct = event_dict.get('home_run', 0) / n_pa
            hbp_pct = event_dict.get('hit_by_pitch', 0) / n_pa
            strikeout_pct = event_dict.get('strikeout', 0) / n_pa
            walk_pct = event_dict.get('walk', 0) / n_pa

            single_cnt = event_dict.get('single', 0)
            double_cnt = event_dict.get('double', 0)
            triple_cnt = event_dict.get('triple', 0)
            hr_cnt = event_dict.get('home_run', 0)
            hbp_cnt = event_dict.get('hit_by_pitch', 0)
            strikeout_cnt = event_dict.get('strikeout', 0)
            walk_cnt = event_dict.get('walk', 0)

            ump_call = player_df['type'].tolist()
            call_types, call_counts = np.unique(ump_call, return_counts=True)
            call_dict = dict(zip(call_types, call_counts))
            n_strikes = call_dict.get('S', 0)
            n_balls = call_dict.get('B', 0)
            strike_pct = n_strikes / (n_strikes + n_balls) if n_strikes + n_balls > 0 else 0
            ball_pct = n_balls / (n_strikes + n_balls) if n_strikes + n_balls > 0 else 0

            avg_vx0 = player_df['vx0'].mean()
            avg_vy0 = player_df['vy0'].mean()
            avg_vz0 = player_df['vz0'].mean()
            avg_ax = player_df['ax'].mean()
            avg_ay = player_df['ay'].mean()
            avg_az = player_df['az'].mean()
            avg_hist_distance = player_df['hit_distance_sc'].mean()
            avg_launch_speed = player_df['launch_speed'].mean()
            avg_launch_angle = player_df['launch_angle'].mean()
            avg_spin_rate = player_df['release_spin_rate'].mean()

            stats_dict = {'n_batters_faced': n_pa,
                          'total_pitch_count': n_strikes + n_balls,
                          'fc_pct': fc_pct,
                          'ff_pct': ff_pct,
                          'sl_pct': sl_pct,
                          'ch_pct': ch_pct,
                          'cu_pct': cu_pct,
                          'si_pct': si_pct,
                          'fs_pct': fs_pct,
                          'ft_pct': ft_pct,
                          'kc_pct': kc_pct,
                          'po_pct': po_pct,
                          'in_pct': in_pct,
                          'sc_pct': sc_pct,
                          'fa_pct': fa_pct,
                          'ep_pct': ep_pct,
                          'kn_pct': kn_pct,
                          'fo_pct': fo_pct,
                          'un_pct': un_pct,
                          'fc_cnt': fc_cnt,
                          'ff_cnt': ff_cnt,
                          'sl_cnt': sl_cnt,
                          'ch_cnt': ch_cnt,
                          'cu_cnt': cu_cnt,
                          'si_cnt': si_cnt,
                          'fs_cnt': fs_cnt,
                          'ft_cnt': ft_cnt,
                          'kc_cnt': kc_cnt,
                          'po_cnt': po_cnt,
                          'in_cnt': in_cnt,
                          'sc_cnt': sc_cnt,
                          'fa_cnt': fa_cnt,
                          'ep_cnt': ep_cnt,
                          'kn_cnt': kn_cnt,
                          'fo_cnt': fo_cnt,
                          'un_cnt': un_cnt,
                          'single_pct': single_pct,
                          'double_pct': double_pct,
                          'triple_pct': triple_pct,
                          'hr_pct': hr_pct,
                          'hit_batter_pct': hbp_pct,
                          'strikeout_pct': strikeout_pct,
                          'walk_pct': walk_pct,
                          'single_cnt': single_cnt,
                          'double_cnt': double_cnt,
                          'triple_cnt': triple_cnt,
                          'hr_cnt': hr_cnt,
                          'hbp_cnt': hbp_cnt,
                          'strikeout_cnt': strikeout_cnt,
                          'walk_cnt': walk_cnt,
                          'n_strikes': n_strikes,
                          'n_balls': n_balls,
                          'strike_pct': strike_pct,
                          'ball_pct': ball_pct,
                          'avg_sz_bot': avg_sz_bot,
                          'avg_sz_top': avg_sz_top,
                          'x_pos_swing_pcts': x_pos_swing_pcts,
                          'z_pos_swing_pcts': z_pos_swing_pcts,
                          'x_pos_hit_pcts': x_pos_hit_pcts,
                          'z_pos_hit_pcts': z_pos_hit_pcts,
                          'x_pos_pitch_pcts': x_pos_pitch_pcts,
                          'z_pos_pitch_pcts': z_pos_pitch_pcts,
                          'avg_vx0': avg_vx0,
                          'avg_vy0': avg_vy0,
                          'avg_vz0': avg_vz0,
                          'avg_ax': avg_ax,
                          'avg_ay': avg_ay,
                          'avg_az': avg_az,
                          'avg_hist_distance': avg_hist_distance,
                          'avg_launch_speed': avg_launch_speed,
                          'avg_launch_angle': avg_launch_angle,
                          'avg_spin_rate': avg_spin_rate,
                          'hit_xpos_res': hit_xpos_res,
                          'hit_ypos_res': hit_ypos_res,
                          }
            for k, v in stats_dict.items():
                if type(v) == np.int64:
                    v = int(v)
                elif type(v) == np.float64:
                    v = float(v)
                stats_dict[k] = v
        else:
            x_pos_swing_pcts = [0 for _ in range(n_xpos_ids)]
            z_pos_swing_pcts = [0 for _ in range(n_zpos_ids)]
            x_pos_hit_pcts = [0 for _ in range(n_xpos_ids)]
            z_pos_hit_pcts = [0 for _ in range(n_zpos_ids)]
            x_pos_pitch_pcts = [0 for _ in range(n_xpos_ids)]
            z_pos_pitch_pcts = [0 for _ in range(n_zpos_ids)]
            hit_xpos_res = [0 for _ in range(n_coord_ids)]
            hit_ypos_res = [0 for _ in range(n_coord_ids)]

            stats_dict = {'n_batters_faced': 0,
                          'total_pitch_count': 0,
                          'fc_pct': 0,
                          'ff_pct': 0,
                          'sl_pct': 0,
                          'ch_pct': 0,
                          'cu_pct': 0,
                          'si_pct': 0,
                          'fs_pct': 0,
                          'ft_pct': 0,
                          'kc_pct': 0,
                          'po_pct': 0,
                          'in_pct': 0,
                          'sc_pct': 0,
                          'fa_pct': 0,
                          'ep_pct': 0,
                          'kn_pct': 0,
                          'fo_pct': 0,
                          'un_pct': 0,
                          'fc_cnt': 0,
                          'ff_cnt': 0,
                          'sl_cnt': 0,
                          'ch_cnt': 0,
                          'cu_cnt': 0,
                          'si_cnt': 0,
                          'fs_cnt': 0,
                          'ft_cnt': 0,
                          'kc_cnt': 0,
                          'po_cnt': 0,
                          'in_cnt': 0,
                          'sc_cnt': 0,
                          'fa_cnt': 0,
                          'ep_cnt': 0,
                          'kn_cnt': 0,
                          'fo_cnt': 0,
                          'un_cnt': 0,
                          'single_pct': 0,
                          'double_pct': 0,
                          'triple_pct': 0,
                          'hr_pct': 0,
                          'hit_batter_pct': 0,
                          'strikeout_pct': 0,
                          'walk_pct': 0,
                          'single_cnt': 0,
                          'double_cnt': 0,
                          'triple_cnt': 0,
                          'hr_cnt': 0,
                          'hbp_cnt': 0,
                          'strikeout_cnt': 0,
                          'walk_cnt': 0,
                          'n_strikes': 0,
                          'n_balls': 0,
                          'strike_pct': 0,
                          'ball_pct': 0,
                          'avg_sz_bot': 0,
                          'avg_sz_top': 0,
                          'x_pos_swing_pcts': x_pos_swing_pcts,
                          'z_pos_swing_pcts': z_pos_swing_pcts,
                          'x_pos_hit_pcts': x_pos_hit_pcts,
                          'z_pos_hit_pcts': z_pos_hit_pcts,
                          'x_pos_pitch_pcts': x_pos_pitch_pcts,
                          'z_pos_pitch_pcts': z_pos_pitch_pcts,
                          'avg_vx0': 0,
                          'avg_vy0': 0,
                          'avg_vz0': 0,
                          'avg_ax': 0,
                          'avg_ay': 0,
                          'avg_az': 0,
                          'avg_hist_distance': 0,
                          'avg_launch_speed': 0,
                          'avg_launch_angle': 0,
                          'avg_spin_rate': 0,
                          'hit_xpos_res': hit_xpos_res,
                          'hit_ypos_res': hit_ypos_res,
                          }

        return stats_dict

    def get_matchup_stats(self, pitcher_id, batter_id, current_year, game_pk=None, ab_number=None, last_n_days=None,
                          game_date=None):
        """
        Get previous matchup statistics between a specified pitcher and batter over a given period of time
        :param pitcher_id: ID of pitcher taking part in the at-bat
        :param batter_id: ID of batter taking part in the at-bat
        :param current_year: year of at-bat for which previous matchups will be queried
        :param game_pk: game of at-bat for which previous matchups will be queried
        :param ab_number: at-bat number of at-bat for which previous matchups will be queried
        :param last_n_days: how many days before given matchup will be queried for historical matchups
        :param game_date: date of at-bat for which previous matchups will be queried
        :return: statistics describing historical matchups
        """
        if game_pk is None and last_n_days is None and ab_number is None and game_date is None:
            query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                                              home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                                              at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                                        from statcast 
                                        where pitcher = ?
                                        and batter = ?
                                        and game_year < ?"""
            query_args = (pitcher_id, batter_id, current_year)
        elif last_n_days is None and ab_number is None and game_date is None:
            query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                              home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                              at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                        from statcast 
                        where pitcher = ?
                        and batter = ?
                        and game_year = ?
                        and game_pk < ?"""
            query_args = (pitcher_id, batter_id, current_year, game_pk)
        else:
            query = """select pitch_type, game_date, release_speed, batter, pitcher, events, game_type, stand, p_throws,
                              home_team, away_team, type, hit_location, bb_type, balls, strikes, game_year, 
                              at_bat_number, sz_top, sz_bot, game_pk, pitch_number, plate_x, plate_z, description,
                                  hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_distance_sc, launch_speed, launch_angle,
                                  release_spin_rate
                        from statcast 
                        where pitcher = ?
                        and batter = ?
                        and game_year = ?
                        and (days_since_2000 between julianday(?) - julianday(?) - ? and julianday(?) - julianday(?) - 1
                        or (game_pk = ? and at_bat_number < ?))"""
            query_args = (pitcher_id, batter_id, current_year, game_date, '2000-01-01', last_n_days, game_date,
                          '2000-01-01', game_pk, ab_number)

        player_season_data = self.query_db(query, query_args)

        attr_names = ['pitch_type', 'game_date', 'release_speed', 'batter', 'pitcher', 'events', 'game_type', 'stand',
                      'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes',
                      'game_year', 'at_bat_number', 'sz_top', 'sz_bot', 'game_pk', 'pitch_number',
                      'plate_x', 'plate_z', 'description', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                      'hit_distance_sc', 'launch_speed', 'launch_angle', 'release_spin_rate']
        player_df = pd.DataFrame(player_season_data, columns=attr_names).fillna(0)
        player_df['plate_x_id'] = player_df['plate_x'].apply(lambda x: decode_xpos(x))
        player_df['plate_z_id'] = player_df['plate_z'].apply(lambda z: decode_zpos(z))

        desc_to_swing_status = {"ball": 0, "called_strike": 0, "blocked_ball": 0, "intent_ball": 0, "hit_by_pitch": 0,
                                "pitchout": 0, "unknown_strike": 0,
                                "foul": 1, "hit_into_play": 1, "swinging_strike": 1, "hit_into_play_no_out": 1,
                                "hit_into_play_score": 1, "foul_tip": 1, "swinging_strike_blocked": 1, "foul_bunt": 1,
                                "missed_bunt": 1, "swinging_pitchout": 1, "pitchout_hit_into_play": 1,
                                "foul_pitchout": 1, "pitchout_hit_into_play_score": 1,
                                "pitchout_hit_into_play_no_out": 1}

        n_xpos_ids = 8
        n_zpos_ids = 9
        n_coord_ids = 10

        if player_df.shape[0] > 0:
            x_pos_swing_res = [0 for _ in range(n_xpos_ids)]
            z_pos_swing_res = [0 for _ in range(n_zpos_ids)]
            x_pos_hit_res = [0 for _ in range(n_xpos_ids)]
            z_pos_hit_res = [0 for _ in range(n_zpos_ids)]
            x_pos_pitches = [0 for _ in range(n_xpos_ids)]
            z_pos_pitches = [0 for _ in range(n_zpos_ids)]
            n_pitches = 0

            plate_x_ids = player_df['plate_x_id'].tolist()
            plate_z_ids = player_df['plate_z_id'].tolist()
            descriptions = player_df['description'].tolist()
            types = player_df['type'].tolist()

            for x_id, z_id, desc, call_type in zip(plate_x_ids, plate_z_ids, descriptions, types):
                # swing_status = desc_to_swing_status[desc]
                swing_status = desc_to_swing_status.get(desc, 0)
                hit_status = 1 if call_type.strip() == 'X' else 0
                x_pos_swing_res[x_id] += swing_status
                z_pos_swing_res[z_id] += swing_status
                x_pos_hit_res[x_id] += hit_status
                z_pos_hit_res[z_id] += hit_status
                x_pos_pitches[x_id] += 1
                z_pos_pitches[z_id] += 1
                n_pitches += 1

            x_pos_swing_pcts = [x / n_pitches for x in x_pos_swing_res]
            z_pos_swing_pcts = [z / n_pitches for z in z_pos_swing_res]
            x_pos_hit_pcts = [x / n_pitches for x in x_pos_hit_res]
            z_pos_hit_pcts = [z / n_pitches for z in z_pos_hit_res]
            x_pos_pitch_pcts = [x / n_pitches for x in x_pos_pitches]
            z_pos_pitch_pcts = [z / n_pitches for z in z_pos_pitches]

            hit_xpos_res = [0 for _ in range(n_coord_ids)]
            hit_ypos_res = [0 for _ in range(n_coord_ids)]
            hit_df = player_df[(player_df['hc_x'] != 0) & (player_df['hc_y'] != 0)][['hc_x', 'hc_y']]
            hit_df['x_id'] = hit_df['hc_x'].apply(lambda x: decode_hit_coord(x))
            hit_df['y_id'] = hit_df['hc_y'].apply(lambda y: decode_hit_coord(y))
            hit_x_ids = hit_df['x_id'].tolist()
            hit_y_ids = hit_df['y_id'].tolist()
            n_hits = len(hit_x_ids)

            for x_id, y_id in zip(hit_x_ids, hit_y_ids):
                hit_xpos_res[x_id] += 1
                hit_ypos_res[y_id] += 1
            hit_xpos_res = [x / n_hits if n_hits > 0 else 0.0 for x in hit_xpos_res]
            hit_ypos_res = [y / n_hits if n_hits > 0 else 0.0 for y in hit_ypos_res]

            thrown_pitches = player_df['pitch_type'].tolist()
            pitch_types, pitch_counts = np.unique(thrown_pitches, return_counts=True)
            pitch_dict = dict(zip(pitch_types, pitch_counts))
            fc_pct = pitch_dict.get('FC', 0) / n_pitches
            ff_pct = pitch_dict.get('FF', 0) / n_pitches
            sl_pct = pitch_dict.get('SL', 0) / n_pitches
            ch_pct = pitch_dict.get('CH', 0) / n_pitches
            cu_pct = pitch_dict.get('CU', 0) / n_pitches
            si_pct = pitch_dict.get('SI', 0) / n_pitches
            fs_pct = pitch_dict.get('FS', 0) / n_pitches
            ft_pct = pitch_dict.get('FT', 0) / n_pitches
            kc_pct = pitch_dict.get('KC', 0) / n_pitches
            po_pct = pitch_dict.get('PO', 0) / n_pitches
            in_pct = pitch_dict.get('IN', 0) / n_pitches
            sc_pct = pitch_dict.get('SC', 0) / n_pitches
            fa_pct = pitch_dict.get('FA', 0) / n_pitches
            ep_pct = pitch_dict.get('EP', 0) / n_pitches
            kn_pct = pitch_dict.get('KN', 0) / n_pitches
            fo_pct = pitch_dict.get('FO', 0) / n_pitches
            un_pct = pitch_dict.get('UN', 0) / n_pitches

            fc_cnt = pitch_dict.get('FC', 0)
            ff_cnt = pitch_dict.get('FF', 0)
            sl_cnt = pitch_dict.get('SL', 0)
            ch_cnt = pitch_dict.get('CH', 0)
            cu_cnt = pitch_dict.get('CU', 0)
            si_cnt = pitch_dict.get('SI', 0)
            fs_cnt = pitch_dict.get('FS', 0)
            ft_cnt = pitch_dict.get('FT', 0)
            kc_cnt = pitch_dict.get('KC', 0)
            po_cnt = pitch_dict.get('PO', 0)
            in_cnt = pitch_dict.get('IN', 0)
            sc_cnt = pitch_dict.get('SC', 0)
            fa_cnt = pitch_dict.get('FA', 0)
            ep_cnt = pitch_dict.get('EP', 0)
            kn_cnt = pitch_dict.get('KN', 0)
            fo_cnt = pitch_dict.get('FO', 0)
            un_cnt = pitch_dict.get('UN', 0)

            sz_bots = [szb for szb in player_df['sz_bot'].tolist() if not szb == 0]
            avg_sz_bot = sum(sz_bots) / len(sz_bots) if not len(sz_bots) == 0 else 0

            sz_tops = [szt for szt in player_df['sz_top'].tolist() if not szt == 0]
            avg_sz_top = sum(sz_tops) / len(sz_tops) if not len(sz_tops) == 0 else 0

            pa_ids = ['{}-{}'.format(game_pk, ab_no) for game_pk, ab_no in zip(player_df['game_pk'].tolist(),
                                                                               player_df['at_bat_number'].tolist())]
            u_pa_ids = list(np.unique(pa_ids))
            n_pa = len(u_pa_ids)
            ab_events = player_df['events'].tolist()
            event_types, event_counts = np.unique(ab_events, return_counts=True)
            event_dict = dict(zip(event_types, event_counts))

            single_pct = event_dict.get('single', 0) / n_pa
            double_pct = event_dict.get('double', 0) / n_pa
            triple_pct = event_dict.get('triple', 0) / n_pa
            hr_pct = event_dict.get('home_run', 0) / n_pa
            hbp_pct = event_dict.get('hit_by_pitch', 0) / n_pa
            strikeout_pct = event_dict.get('strikeout', 0) / n_pa
            walk_pct = event_dict.get('walk', 0) / n_pa

            single_cnt = event_dict.get('single', 0)
            double_cnt = event_dict.get('double', 0)
            triple_cnt = event_dict.get('triple', 0)
            hr_cnt = event_dict.get('home_run', 0)
            hbp_cnt = event_dict.get('hit_by_pitch', 0)
            strikeout_cnt = event_dict.get('strikeout', 0)
            walk_cnt = event_dict.get('walk', 0)

            ump_call = player_df['type'].tolist()
            call_types, call_counts = np.unique(ump_call, return_counts=True)
            call_dict = dict(zip(call_types, call_counts))
            n_strikes = call_dict.get('S', 0)
            n_balls = call_dict.get('B', 0)
            strike_pct = n_strikes / (n_strikes + n_balls) if n_strikes + n_balls > 0 else 0
            ball_pct = n_balls / (n_strikes + n_balls) if n_strikes + n_balls > 0 else 0

            avg_vx0 = player_df['vx0'].mean()
            avg_vy0 = player_df['vy0'].mean()
            avg_vz0 = player_df['vz0'].mean()
            avg_ax = player_df['ax'].mean()
            avg_ay = player_df['ay'].mean()
            avg_az = player_df['az'].mean()
            avg_hist_distance = player_df['hit_distance_sc'].mean()
            avg_launch_speed = player_df['launch_speed'].mean()
            avg_launch_angle = player_df['launch_angle'].mean()
            avg_spin_rate = player_df['release_spin_rate'].mean()

            stats_dict = {'n_batters_faced': n_pa,
                          'total_pitch_count': n_strikes + n_balls,
                          'fc_pct': fc_pct,
                          'ff_pct': ff_pct,
                          'sl_pct': sl_pct,
                          'ch_pct': ch_pct,
                          'cu_pct': cu_pct,
                          'si_pct': si_pct,
                          'fs_pct': fs_pct,
                          'ft_pct': ft_pct,
                          'kc_pct': kc_pct,
                          'po_pct': po_pct,
                          'in_pct': in_pct,
                          'sc_pct': sc_pct,
                          'fa_pct': fa_pct,
                          'ep_pct': ep_pct,
                          'kn_pct': kn_pct,
                          'fo_pct': fo_pct,
                          'un_pct': un_pct,
                          'fc_cnt': fc_cnt,
                          'ff_cnt': ff_cnt,
                          'sl_cnt': sl_cnt,
                          'ch_cnt': ch_cnt,
                          'cu_cnt': cu_cnt,
                          'si_cnt': si_cnt,
                          'fs_cnt': fs_cnt,
                          'ft_cnt': ft_cnt,
                          'kc_cnt': kc_cnt,
                          'po_cnt': po_cnt,
                          'in_cnt': in_cnt,
                          'sc_cnt': sc_cnt,
                          'fa_cnt': fa_cnt,
                          'ep_cnt': ep_cnt,
                          'kn_cnt': kn_cnt,
                          'fo_cnt': fo_cnt,
                          'un_cnt': un_cnt,
                          'single_pct': single_pct,
                          'double_pct': double_pct,
                          'triple_pct': triple_pct,
                          'hr_pct': hr_pct,
                          'hit_batter_pct': hbp_pct,
                          'strikeout_pct': strikeout_pct,
                          'walk_pct': walk_pct,
                          'single_cnt': single_cnt,
                          'double_cnt': double_cnt,
                          'triple_cnt': triple_cnt,
                          'hr_cnt': hr_cnt,
                          'hbp_cnt': hbp_cnt,
                          'strikeout_cnt': strikeout_cnt,
                          'walk_cnt': walk_cnt,
                          'n_strikes': n_strikes,
                          'n_balls': n_balls,
                          'strike_pct': strike_pct,
                          'ball_pct': ball_pct,
                          'avg_sz_bot': avg_sz_bot,
                          'avg_sz_top': avg_sz_top,
                          'x_pos_swing_pcts': x_pos_swing_pcts,
                          'z_pos_swing_pcts': z_pos_swing_pcts,
                          'x_pos_hit_pcts': x_pos_hit_pcts,
                          'z_pos_hit_pcts': z_pos_hit_pcts,
                          'x_pos_pitch_pcts': x_pos_pitch_pcts,
                          'z_pos_pitch_pcts': z_pos_pitch_pcts,
                          'avg_vx0': avg_vx0,
                          'avg_vy0': avg_vy0,
                          'avg_vz0': avg_vz0,
                          'avg_ax': avg_ax,
                          'avg_ay': avg_ay,
                          'avg_az': avg_az,
                          'avg_hist_distance': avg_hist_distance,
                          'avg_launch_speed': avg_launch_speed,
                          'avg_launch_angle': avg_launch_angle,
                          'avg_spin_rate': avg_spin_rate,
                          'hit_xpos_res': hit_xpos_res,
                          'hit_ypos_res': hit_ypos_res,
                          }
            for k, v in stats_dict.items():
                if type(v) == np.int64:
                    v = int(v)
                elif type(v) == np.float64:
                    v = float(v)
                stats_dict[k] = v
        else:
            x_pos_swing_pcts = [0 for _ in range(n_xpos_ids)]
            z_pos_swing_pcts = [0 for _ in range(n_zpos_ids)]
            x_pos_hit_pcts = [0 for _ in range(n_xpos_ids)]
            z_pos_hit_pcts = [0 for _ in range(n_zpos_ids)]
            x_pos_pitch_pcts = [0 for _ in range(n_xpos_ids)]
            z_pos_pitch_pcts = [0 for _ in range(n_zpos_ids)]
            hit_xpos_res = [0 for _ in range(n_coord_ids)]
            hit_ypos_res = [0 for _ in range(n_coord_ids)]

            stats_dict = {'n_batters_faced': 0,
                          'total_pitch_count': 0,
                          'fc_pct': 0,
                          'ff_pct': 0,
                          'sl_pct': 0,
                          'ch_pct': 0,
                          'cu_pct': 0,
                          'si_pct': 0,
                          'fs_pct': 0,
                          'ft_pct': 0,
                          'kc_pct': 0,
                          'po_pct': 0,
                          'in_pct': 0,
                          'sc_pct': 0,
                          'fa_pct': 0,
                          'ep_pct': 0,
                          'kn_pct': 0,
                          'fo_pct': 0,
                          'un_pct': 0,
                          'fc_cnt': 0,
                          'ff_cnt': 0,
                          'sl_cnt': 0,
                          'ch_cnt': 0,
                          'cu_cnt': 0,
                          'si_cnt': 0,
                          'fs_cnt': 0,
                          'ft_cnt': 0,
                          'kc_cnt': 0,
                          'po_cnt': 0,
                          'in_cnt': 0,
                          'sc_cnt': 0,
                          'fa_cnt': 0,
                          'ep_cnt': 0,
                          'kn_cnt': 0,
                          'fo_cnt': 0,
                          'un_cnt': 0,
                          'single_pct': 0,
                          'double_pct': 0,
                          'triple_pct': 0,
                          'hr_pct': 0,
                          'hit_batter_pct': 0,
                          'strikeout_pct': 0,
                          'walk_pct': 0,
                          'single_cnt': 0,
                          'double_cnt': 0,
                          'triple_cnt': 0,
                          'hr_cnt': 0,
                          'hbp_cnt': 0,
                          'strikeout_cnt': 0,
                          'walk_cnt': 0,
                          'n_strikes': 0,
                          'n_balls': 0,
                          'strike_pct': 0,
                          'ball_pct': 0,
                          'avg_sz_bot': 0,
                          'avg_sz_top': 0,
                          'x_pos_swing_pcts': x_pos_swing_pcts,
                          'z_pos_swing_pcts': z_pos_swing_pcts,
                          'x_pos_hit_pcts': x_pos_hit_pcts,
                          'z_pos_hit_pcts': z_pos_hit_pcts,
                          'x_pos_pitch_pcts': x_pos_pitch_pcts,
                          'z_pos_pitch_pcts': z_pos_pitch_pcts,
                          'avg_vx0': 0,
                          'avg_vy0': 0,
                          'avg_vz0': 0,
                          'avg_ax': 0,
                          'avg_ay': 0,
                          'avg_az': 0,
                          'avg_hist_distance': 0,
                          'avg_launch_speed': 0,
                          'avg_launch_angle': 0,
                          'avg_spin_rate': 0,
                          'hit_xpos_res': hit_xpos_res,
                          'hit_ypos_res': hit_ypos_res,
                          }

        return stats_dict

    def get_next_gamestate(self, game_pk, ab_number):
        """
        Get the gamestate at the beginning of the next at-bat
        :param game_pk: game being queried
        :param ab_number: current at-bat number. Gamestate at beginning of at-bat ab_number+1 will be returned.
        :return:
        """
        query = """select home_score, away_score, bat_score, fld_score, on_1b, on_2b, on_3b, batter, pitcher,
                            inning, inning_topbot, outs_when_up
                   from statcast
                   where game_pk = ? and at_bat_number = ?
                   order by pitch_number asc limit 1"""
        query_args = (game_pk, ab_number + 1)
        gamestate_data = self.query_db(query, query_args)

        if len(gamestate_data) > 0:
            gamestate_data = gamestate_data[0]
            next_gamestate = {'home_score': gamestate_data[0],
                              'away_score': gamestate_data[1],
                              'bat_score': gamestate_data[2],
                              'fld_score': gamestate_data[3],
                              'on_1b': gamestate_data[4],
                              'on_2b': gamestate_data[5],
                              'on_3b': gamestate_data[6],
                              'batter': gamestate_data[7],
                              'pitcher': gamestate_data[8],
                              'inning': gamestate_data[9],
                              'inning_topbot': gamestate_data[10],
                              'outs_when_up': gamestate_data[11],
                              }
        else:
            next_gamestate = {'home_score': 'N/A',
                              'away_score': 'N/A',
                              'bat_score': 'N/A',
                              'fld_score': 'N/A',
                              'on_1b': 'N/A',
                              'on_2b': 'N/A',
                              'on_3b': 'N/A',
                              'batter': 'N/A',
                              'pitcher': 'N/A',
                              'inning': 'N/A',
                              'inning_topbot': 'N/A',
                              'outs_when_up': 'N/A',
                              }

        return next_gamestate
