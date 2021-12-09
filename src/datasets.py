__author__ = 'Connor Heaton'

import os
import json
import math
import torch
import random

from torch.utils.data import Dataset

from vocabularies import GeneralGamestateDeltaVocab


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


def read_file_lines(fp, bad_data_fps=None):
    if bad_data_fps is None:
        bad_data_fps = []
    lines = []
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '' and line not in bad_data_fps:
                lines.append(line)

    return lines


def bucketize(data, min, max, n_buckets):
    new_data = []
    bucket_size = (max - min) / n_buckets
    for x in data:
        try:
            if x < min:
                x = min
            elif x > max:
                x = max

            x += abs(min)
            bucket = int(x // bucket_size)
            if bucket < 0:
                bucket = 0
            elif bucket >= n_buckets:
                bucket = n_buckets - 1

            new_data.append(bucket)
        except:
            new_data.append(0)

    return new_data


def faux_normalize(x, min_mag=-10.0, max_mag=10.0):
    if x < min_mag:
        x = min_mag
    elif x > max_mag:
        x = max_mag

    x = x / max(abs(min_mag), max_mag)
    return x


def fill_null(data, fill_val, norm_val=1.0):
    data = [x / norm_val if x is not None else fill_val for x in data]
    return data


def parse_pitches(pitch_j, parse_pos, norm_vals=None):
    if norm_vals is None:
        norm_vals = {}
    pitch_info = [[p['balls'], p['strikes'], p['pitch_type'], p['release_speed'],
                   p['plate_x'], p['plate_z'], p['release_pos_x'], p['release_pos_y'],
                   p['release_pos_z'], p['release_spin_rate'], p['release_extension'],
                   p['hc_x'], p['hc_y'], p['vx0'], p['vy0'], p['vz0'], p['ax'], p['ay'], p['az'],
                   p['hit_distance_sc'], p['launch_speed'], p['launch_angle']] for p in pitch_j]
    balls, strikes, pitch_type, pitch_speed, plate_x, plate_z, release_x, release_y, release_z, spin_rate, extension, hc_x, hc_y, vx0, vy0, vz0, ax, ay, az, hit_dist, launch_speed, launch_angle = map(
        list, zip(*pitch_info))

    pitch_speed = fill_null(pitch_speed, 0, norm_val=norm_vals.get('release_speed', 1.0))
    release_x = fill_null(release_x, 0, norm_val=norm_vals.get('release_pos_x', 1.0))
    release_y = fill_null(release_y, 0, norm_val=norm_vals.get('release_pos_y', 1.0))
    release_z = fill_null(release_z, 0, norm_val=norm_vals.get('release_pos_z', 1.0))
    spin_rate = fill_null(spin_rate, 0, norm_val=norm_vals.get('release_spin_rate', 1.0))
    extension = fill_null(extension, 0, norm_val=norm_vals.get('release_extension', 1.0))
    hc_x = fill_null(hc_x, 0, norm_val=norm_vals.get('hc_x', 1.0))
    hc_y = fill_null(hc_y, 0, norm_val=norm_vals.get('hc_y', 1.0))
    vx0 = fill_null(vx0, 0, norm_val=norm_vals.get('vx0', 1.0))
    vy0 = fill_null(vy0, 0, norm_val=norm_vals.get('vy0', 1.0))
    vz0 = fill_null(vz0, 0, norm_val=norm_vals.get('vz0', 1.0))
    ax = fill_null(ax, 0, norm_val=norm_vals.get('ax', 1.0))
    ay = fill_null(ay, 0, norm_val=norm_vals.get('ay', 1.0))
    az = fill_null(az, 0, norm_val=norm_vals.get('az', 1.0))
    hit_dist = fill_null(hit_dist, 0, norm_val=norm_vals.get('hit_distance_sc', 1.0))
    launch_speed = fill_null(launch_speed, 0, norm_val=norm_vals.get('launch_speed', 1.0))
    launch_angle = fill_null(launch_angle, 0, norm_val=norm_vals.get('launch_angle', 1.0))

    rv_inputs = [pitch_speed, release_x, release_y, release_z, spin_rate, extension, hc_x, hc_y, vx0, vy0, vz0,
                 ax, ay, az, hit_dist, launch_speed, launch_angle]
    rv_inputs = list(zip(*rv_inputs))

    if parse_pos:
        plate_x = bucketize(plate_x, min=-2, max=2, n_buckets=8)
        plate_z = bucketize(plate_z, min=0, max=4, n_buckets=9)

    return balls, strikes, pitch_type, plate_x, plate_z, rv_inputs


def parse_w_sign(s):
    if type(s) == int:
        pass
    elif s in [None]:  # , 'N/A'
        s = 0
    elif s[0] in ['+', '-']:
        s = int(s[1:])
    else:
        s = int(s)

    return s


def compare_states(curr_state, last_state, fill_value):
    attrs_to_compare = ['balls', 'strikes', 'outs', 'on_1b', 'on_2b', 'on_3b', 'score']
    delta = {}
    if last_state is None:
        for attr_name in attrs_to_compare:
            delta[attr_name] = fill_value
    else:
        for attr_name in attrs_to_compare:
            curr_val = parse_w_sign(curr_state.get(attr_name, 0))
            last_val = parse_w_sign(last_state.get(attr_name, 0))
            change_val = curr_val - last_val
            if change_val > 0:
                change_str = '+{}'.format(change_val)
            else:
                change_str = str(change_val)
            delta[attr_name] = change_str

    if delta['balls'][0] == '-':
        delta['strikes'] = '-1'
    if delta['strikes'][0] == '-':
        delta['balls'] = '-1'

    return delta, curr_state


def find_state_deltas_v2(n_balls, n_strikes, outs_when_up, inning_topbot, next_state, batter_score,
                         on_1b, on_2b, on_3b, last_state=None):
    deltas = []
    this_last_state = last_state
    for p_idx, (balls, strikes) in enumerate(zip(n_balls, n_strikes)):
        p_state_dict = {'balls': balls,
                        'strikes': strikes,
                        'outs': outs_when_up,
                        'on_1b': on_1b,
                        'on_2b': on_2b,
                        'on_3b': on_3b,
                        'score': batter_score}
        p_delta, this_last_state = compare_states(p_state_dict, this_last_state, '[BOI]')
        deltas.append(p_delta)

    p_state_dict = {}
    if next_state['home_score'] == 'N/A':
        p_state_dict['balls'] = 0
        p_state_dict['strikes'] = 0
        p_state_dict['on_1b'] = on_1b
        p_state_dict['on_2b'] = on_2b
        p_state_dict['on_3b'] = on_3b
        p_state_dict['outs'] = outs_when_up
        p_state_dict['score'] = batter_score + 1
    else:
        p_state_dict['balls'] = 0
        p_state_dict['strikes'] = 0
        p_state_dict['on_1b'] = 0 if next_state['on_1b'] in [None, 0] else 1
        p_state_dict['on_2b'] = 0 if next_state['on_2b'] in [None, 0] else 1
        p_state_dict['on_3b'] = 0 if next_state['on_3b'] in [None, 0] else 1
        p_state_dict['outs'] = next_state['outs_when_up'] \
            if next_state['inning_topbot'].lower() == inning_topbot.lower() else 3
        p_state_dict['score'] = next_state['bat_score'] \
            if next_state['inning_topbot'].lower() == inning_topbot.lower() else next_state['fld_score']

    p_delta, this_last_state = compare_states(p_state_dict, this_last_state, '[BOI]')
    deltas.append(p_delta)

    return deltas, this_last_state


def parse_json(j, j_type, max_value_data=None):
    if j_type == 'pitcher':
        avoid_keys = ['__id__', 'throws', 'first_name', 'last_name']
        handedness = j['throws']
    elif j_type == 'batter':
        avoid_keys = ['__id__', 'stand', 'first_name', 'last_name']
        handedness = j['stand']
    else:
        avoid_keys = []
        handedness = None

    if max_value_data is not None:
        type_max_value_data = max_value_data[j_type]
    else:
        type_max_value_data = {}

    data = {'handedness': handedness}
    for k1, v1 in j.items():
        k1_norm_vals = type_max_value_data.get(k1, {})
        if k1 not in avoid_keys:
            k1_data = []
            for k2, v2 in v1.items():
                if type(v2) == list:
                    k2_norm_val = k1_norm_vals.get(k2, [1.0])
                    k1_data.extend(
                        [float(item_val) / norm_val if not math.isnan(item_val) and not item_val == math.inf else 1.0
                         for item_val, norm_val in zip(v2, k2_norm_val)])
                else:
                    if v2 == math.inf:
                        new_v2 = 1.0
                    elif math.isnan(v2):
                        new_v2 = 0.0
                    else:
                        k2_norm_val = k1_norm_vals.get(k2, 1.0)
                        new_v2 = v2 / k2_norm_val if k2_norm_val != 0 else v2
                    k1_data.append(float(new_v2))
            data[k1] = k1_data

    if math.nan in data:
        input('j: {}'.format(j))

    return data


class PlayerFormDataset(Dataset):
    def __init__(self, args, mode, gamestate_vocab=None, ab_data_cache=None, memory_cache=None):
        self.args = args
        self.mode = mode if mode in ['train', 'test', 'apply'] else 'dev'
        self.gamestate_vocab = gamestate_vocab
        memory_cache = {} if memory_cache is None else memory_cache
        torch.manual_seed(self.args.seed)

        # parms specific to this dataset
        self.player_type = getattr(self.args, 'player_type', 'batter')
        self.form_ab_window_size = getattr(self.args, 'form_ab_window_size', 25)
        self.min_form_ab_window_size = getattr(self.args, 'min_form_ab_window_size', 20)
        self.min_ab_to_be_included_in_dataset = getattr(self.args, 'min_ab_to_be_included_in_dataset', 200)
        self.ab_data_dir = getattr(self.args, 'ab_data', '/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v7')
        self.bad_data_fps = getattr(self.args, 'bad_data_fps', [])
        self.do_mem_cache = getattr(self.args, 'do_mem_cache', False)

        self.batter_data_scopes_to_use = getattr(self.args, 'batter_data_scopes_to_use',
                                                 ['career', 'season', 'last15', 'this_game'])
        self.pitcher_data_scopes_to_use = getattr(self.args, 'pitcher_data_scopes_to_use',
                                                  ['career', 'season', 'last15', 'this_game'])
        self.matchup_data_scopes_to_use = getattr(self.args, 'matchup_data_scopes_to_use',
                                                  ['career', 'season', 'this_game'])

        self.raw_pitcher_data_dim = getattr(self.args, 'raw_pitcher_data_dim', 274)
        self.raw_batter_data_dim = getattr(self.args, 'raw_batter_data_dim', 274)
        self.raw_matchup_data_dim = getattr(self.args, 'raw_matchup_data_dim', 137)

        self.distribution_based_player_sampling_prob = getattr(self.args,
                                                               'distribution_based_player_sampling_prob', 0.25)
        self.full_ab_window_size_prob = getattr(self.args, 'full_ab_window_size_prob', 1.0)

        print('PlayerFormDataset batter_data_scopes_to_use: {}'.format(self.batter_data_scopes_to_use))
        print('PlayerFormDataset pitcher_data_scopes_to_use: {}'.format(self.pitcher_data_scopes_to_use))

        # other parms now
        self.max_seq_len = getattr(self.args, 'max_seq_len', 200)
        self.max_view_len = getattr(self.args, 'max_view_len', 125)
        self.view_size = getattr(self.args, 'view_size', 15)
        self.n_views = getattr(self.args, 'n_views', 2)
        self.min_view_step_size = getattr(self.args, 'min_view_step_size', 1)
        self.max_view_step_size = getattr(self.args, 'max_view_step_size', 5)

        self.career_data_dir = getattr(self.args, 'career_data',
                                       '/home/czh/sata1/learning_player_form/player_career_data')
        self.whole_game_record_dir = getattr(self.args, 'whole_game_record_dir',
                                             '/home/czh/sata1/learning_player_form/whole_game_records')
        self.parse_plate_pos_to_id = getattr(self.args, 'parse_plate_pos_to_id', False)

        pitch_type_config_fp = getattr(self.args, 'pitch_type_config_fp', '../config/pitch_type_id_mapping.json')
        player_bio_info_fp = getattr(self.args, 'player_bio_info_fp', '../config/statcast_id_to_bio_info.json')
        player_id_map_fp = getattr(self.args, 'player_id_map_fp', '../config/all_player_id_mapping.json')
        team_stadiums_fp = getattr(self.args, 'team_stadiums_fp', '../config/team_stadiums.json')
        player_pos_source = getattr(self.args, 'player_pos_source', 'mlb')

        record_norm_values_fp = os.path.join(self.whole_game_record_dir,
                                             'game_event_splits', 'stats', 'max_values.json')
        self.player_pos_key = '{}_pos'.format(player_pos_source)
        self.player_pos_id_map = json.load(open(getattr(self.args, '{}_id_map_fp'.format(self.player_pos_key),
                                                        '../config/{}_mapping.json'.format(self.player_pos_key))))

        self.record_norm_values = json.load(open(record_norm_values_fp))
        self.pitch_type_mapping = json.load(open(pitch_type_config_fp))
        self.player_id_map = json.load(open(player_id_map_fp), object_hook=jsonKeys2int)
        self.player_bio_info_mapping = json.load(open(player_bio_info_fp), object_hook=jsonKeys2int)
        self.team_stadium_map = json.load(open(team_stadiums_fp))

        self.handedness_id_map = {'L': 0, 'R': 1}

        if self.mode != 'apply':
            # class objects
            self.items_by_player = []
            self.ab_data_cache = ab_data_cache if ab_data_cache is not None else {}

            splits_to_load = ['train', 'dev', 'test']
            print('PlayerFormDataset Loading initial data...')
            for split in splits_to_load:
                print('Loading {} split...'.format(split))
                player_split_fp = os.path.join(self.career_data_dir, '{}-splits'.format(self.player_type),
                                               '{}.txt'.format(split))
                player_type_career_data_dir = os.path.join(self.career_data_dir, self.player_type)
                print('\tplayer_split_fp: {}'.format(player_split_fp))
                print('\tplayer_type_career_data_dir: {}'.format(player_type_career_data_dir))
                self.load_initial_data(player_split_fp, player_type_career_data_dir)

            self.player_items, self.player_item_counts = map(list, zip(*self.items_by_player))

            self.n_player_items = sum(self.player_item_counts)
            self.n_players = len(self.player_item_counts)

            print('# players: {}'.format(self.n_players))
            print('# player records: {}'.format(self.n_player_items))
        else:
            print('$$$$ PlayerFormDataset in apply mode $$$$')
            self.memory_cache = memory_cache

        if self.gamestate_vocab is None:
            gamestate_vocab_bos_inning_no = getattr(self.args, 'gamestate_vocab_bos_inning_no', True)
            gamestate_vocab_bos_score_diff = getattr(self.args, 'gamestate_vocab_bos_score_diff', True)
            gamestate_vocab_use_balls_strikes = getattr(self.args, 'gamestate_vocab_use_balls_strikes', True)
            gamestate_vocab_use_base_occupancy = getattr(self.args, 'gamestate_vocab_use_base_occupancy', True)
            gamestate_vocab_use_inning_no = getattr(self.args, 'gamestate_vocab_use_inning_no', False)
            gamestate_vocab_use_inning_topbot = getattr(self.args, 'gamestate_vocab_use_inning_topbot', False)
            gamestate_vocab_use_score_diff = getattr(self.args, 'gamestate_vocab_use_score_diff', True)
            gamestate_vocab_use_outs = getattr(self.args, 'gamestate_vocab_use_outs', True)
            gamestate_n_innings = getattr(self.args, 'gamestate_n_innings', 10)
            gamestate_max_score_diff = getattr(self.args, 'gamestate_max_score_diff', 6)
            self.gamestate_vocab = GeneralGamestateDeltaVocab(bos_inning_no=gamestate_vocab_bos_inning_no,
                                                              max_inning_no=gamestate_n_innings,
                                                              bos_score_diff=gamestate_vocab_bos_score_diff,
                                                              bos_max_score_diff=gamestate_max_score_diff,
                                                              balls_delta=gamestate_vocab_use_balls_strikes,
                                                              strikes_delta=gamestate_vocab_use_balls_strikes,
                                                              outs_delta=gamestate_vocab_use_outs,
                                                              score_delta=gamestate_vocab_use_score_diff,
                                                              base_occ_delta=gamestate_vocab_use_base_occupancy,
                                                              )

    def __len__(self):
        return 50000 if self.mode == 'train' else 5000

    def __getitem__(self, idx):
        # Get original item
        item = self.get_raw_item()
        # Make multiple views from original item
        item = self.distort_item(item)

        return item

    def load_initial_data(self, split_fp, career_data_dir):
        with open(split_fp, 'r') as f:
            for line in f:
                line = line.strip()
                if not line == '':
                    player_data_fp = os.path.join(career_data_dir, line)
                    player_ab_files = read_file_lines(player_data_fp, self.bad_data_fps)

                    if len(player_ab_files) >= self.min_ab_to_be_included_in_dataset:
                        n_possible_items = max(1, len(player_ab_files) - self.form_ab_window_size + 1)
                        self.items_by_player.append([player_ab_files, n_possible_items])

    def get_raw_item(self):
        """
        Parse a record
        :return: parsed record
        """
        if random.random() <= self.distribution_based_player_sampling_prob:
            player_files = random.choices(self.player_items, weights=self.player_item_counts, k=1)[0]
        else:
            player_files = random.choices(self.player_items, weights=None, k=1)[0]

        n_player_files = len(player_files)
        file_start_idx = random.randint(0, max(n_player_files - self.form_ab_window_size + 1, 1))

        if random.random() <= self.full_ab_window_size_prob:
            file_end_idx = file_start_idx + self.form_ab_window_size
        else:
            this_ab_window_size = random.randint(self.min_form_ab_window_size, self.form_ab_window_size)
            file_end_idx = file_start_idx + this_ab_window_size

        selected_files = player_files[file_start_idx:file_end_idx]

        raw_data = self.parse_player_file_set(selected_files)
        raw_data['n_player_files'] = torch.tensor([n_player_files])
        raw_data['sample_file_start_idx'] = torch.tensor([file_start_idx])
        raw_data['sample_file_end_idx'] = torch.tensor([file_end_idx])

        return raw_data

    def distort_item(self, item):
        """
        Create as many views from item that args constitute
        :param item: record from which to make views
        :return: multi-view record
        """
        curr_ab_idx = 0
        raw_views = []
        for view_no in range(self.n_views):
            ab_start_no = curr_ab_idx
            ab_end_no = ab_start_no + self.view_size
            t_start_idx = sum(item['ab_lengths'][:ab_start_no])
            t_end_idx = sum(item['ab_lengths'][:ab_end_no])
            curr_ab_idx += random.randint(self.min_view_step_size, self.max_view_step_size)

            this_view = self.make_view(item, ab_start_no=ab_start_no, ab_end_no=ab_end_no,
                                       t_start_idx=t_start_idx, t_end_idx=t_end_idx)
            raw_views.append(this_view)

        views = {k: torch.cat([rv[k] for rv in raw_views], dim=0) for k in item.keys()}
        return views

    def make_view(self, item, ab_start_no, ab_end_no, t_start_idx, t_end_idx):
        """
        Make a view of the item using given parameters
        :param item: original record from which to derive a view
        :param ab_start_no: starting at-bat index for view
        :param ab_end_no: ending at-bat index for view
        :param t_start_idx: starting pitch index for view
        :param t_end_idx: ending pitch index for view
        :return: a view derived from the given item
        """
        view = {}
        for k, v in item.items():
            if v.shape[0] == self.form_ab_window_size:
                view[k] = v[ab_start_no:ab_end_no].unsqueeze(0)
            else:
                view_data = v[t_start_idx:t_end_idx]
                if view_data.shape[0] >= self.max_view_len:
                    view_data = view_data[:self.max_view_len]
                elif view_data.shape[0] < self.max_view_len:
                    n_pad = self.max_view_len - view_data.shape[0]

                    if k == 'my_src_pad_mask':
                        v_pad = torch.zeros(n_pad, dtype=view_data.dtype)
                    else:
                        if len(view_data.shape) == 1:
                            v_pad = torch.ones(n_pad, dtype=view_data.dtype)
                        else:
                            v_pad = torch.ones(n_pad, view_data.shape[1], dtype=view_data.dtype)
                    view_data = torch.cat((view_data, v_pad), dim=0)

                view_data = view_data.unsqueeze(0)
                view[k] = view_data

        return view

    def parse_player_file_set(self, file_set):
        """
        Parse all at-bat files in a given set
        :param file_set: list of filepaths to parse
        :return: JSON object corresponding to all files in file set
        """
        tensor_values = ['inning', 'inning_ids', 'ab_number', 'ab_number_ids', 'state_delta_ids',  # 'ab_lengths',
                         'pitch_types', 'plate_x', 'plate_z', 'pitcher_inputs', 'pitcher_id', 'batter_inputs',
                         'batter_id', 'matchup_inputs', 'pitcher_pos_ids', 'batter_pos_ids', 'rv_pitch_data',
                         'game_year', 'game_pk', 'stadium_ids', 'pitcher_handedness_ids', 'batter_handedness_ids']
        player_data = {}

        for fp in file_set:
            if self.mode == 'apply':
                if self.do_mem_cache:
                    if self.memory_cache.get(fp, None) is not None:
                        # print('** cache hit **')
                        fp_data = self.memory_cache[fp]
                    else:
                        # print('** cache miss **')
                        fp_data = self.parse_item_fp(fp)
                        self.memory_cache[fp] = fp_data
                else:
                    fp_data = self.parse_item_fp(fp)
            else:
                if self.do_mem_cache:
                    if self.ab_data_cache.get(fp, None) is not None:
                        fp_data = self.ab_data_cache[fp]
                    else:
                        fp_data = self.parse_item_fp(fp)
                        self.ab_data_cache[fp] = fp_data
                else:
                    fp_data = self.parse_item_fp(fp)

            for k, v in fp_data.items():
                old_v = player_data.get(k, None)
                if old_v is None:
                    if k in tensor_values:
                        new_v = v
                    else:
                        new_v = [v]
                else:
                    if k in tensor_values:
                        new_v = torch.cat([old_v, v], dim=0)
                    else:
                        new_v = old_v[:]
                        new_v.append(v)

                player_data[k] = new_v

        pitch_numbers = []
        record_ab_numbers = []
        for ab_idx, ab_len in enumerate(player_data.get('ab_lengths', [])):
            these_pitch_numbers = [i for i in range(ab_len)]
            these_ab_numbers = [ab_idx for _ in range(ab_len)]

            pitch_numbers.extend(these_pitch_numbers)
            record_ab_numbers.extend(these_ab_numbers)

        player_data['ab_lengths'] = torch.tensor(player_data.get('ab_lengths', []), dtype=torch.long)
        player_data['pitch_numbers'] = torch.tensor(pitch_numbers, dtype=torch.long)
        player_data['record_ab_numbers'] = torch.tensor(record_ab_numbers, dtype=torch.long)

        player_data['game_pk'] = torch.tensor([], dtype=torch.long) if player_data.get('game_pk', None) is None else player_data.get('game_pk', [])
        player_data['game_year'] = torch.tensor([], dtype=torch.long) if player_data.get('game_year', None) is None else player_data.get('game_year', [])
        player_data['inning'] = torch.tensor([], dtype=torch.long) if player_data.get('inning', None) is None else player_data.get('inning', [])
        player_data['inning_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('inning_ids', None) is None else player_data.get('inning_ids', [])
        player_data['stadium_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('stadium_ids', None) is None else player_data.get('stadium_ids', [])
        player_data['ab_number'] = torch.tensor([], dtype=torch.long) if player_data.get('ab_number', None) is None else player_data.get('ab_number', [])
        player_data['ab_number_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('ab_number_ids', None) is None else player_data.get('ab_number_ids', [])
        player_data['state_delta_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('state_delta_ids', None) is None else player_data.get('state_delta_ids', [])
        player_data['pitch_types'] = torch.tensor([], dtype=torch.long) if player_data.get('pitch_types', None) is None else player_data.get('pitch_types', [])
        plate_pos_dtype = torch.long if self.parse_plate_pos_to_id else torch.float
        player_data['plate_x'] = torch.tensor([], dtype=plate_pos_dtype) if player_data.get('plate_x', None) is None else player_data.get('plate_x', [])
        player_data['plate_z'] = torch.tensor([], dtype=plate_pos_dtype) if player_data.get('plate_z', None) is None else player_data.get('plate_z', [])
        player_data['pitcher_inputs'] = torch.tensor([[0.0 for _ in range(self.raw_pitcher_data_dim)]]) if player_data.get('pitcher_inputs', None) is None else player_data.get('pitcher_inputs', [])
        player_data['pitcher_id'] = torch.tensor([], dtype=torch.long) if player_data.get('pitcher_id', None) is None else player_data.get('pitcher_id', [])
        player_data['batter_inputs'] = torch.tensor([[0.0 for _ in range(self.raw_batter_data_dim)]]) if player_data.get('batter_inputs', None) is None else player_data.get('batter_inputs', [])
        player_data['batter_id'] = torch.tensor([], dtype=torch.long) if player_data.get('batter_id', None) is None else player_data.get('batter_id', [])
        player_data['matchup_inputs'] = torch.tensor([[0.0 for _ in range(self.raw_matchup_data_dim)]]) if player_data.get('matchup_inputs', None) is None else player_data.get('matchup_inputs', [])
        player_data['pitcher_pos_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('pitcher_pos_ids', None) is None else player_data.get('pitcher_pos_ids', [])
        player_data['batter_pos_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('batter_pos_ids', None) is None else player_data.get('batter_pos_ids', [])
        player_data['rv_pitch_data'] = torch.tensor([[0.0 for _ in range(17)]]) if player_data.get('rv_pitch_data', None) is None else player_data.get('rv_pitch_data', [])
        player_data['pitcher_handedness_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('pitcher_handedness_ids', None) is None else player_data.get('pitcher_handedness_ids', [])
        player_data['batter_handedness_ids'] = torch.tensor([], dtype=torch.long) if player_data.get('batter_handedness_ids', None) is None else player_data.get('batter_handedness_ids', [])

        player_data = self.finalize_player_data(player_data)
        return player_data

    def parse_item_fp(self, item_fp):
        """
        Parse data at given filepath, select only desired information from object
        :param item_fp: item filepath to parse
        :return: at-bat JSON record
        """
        item_intermediate_data = self.parse_item_intermediate(item_fp)

        game_pk = item_intermediate_data['game_pk']
        game_year = item_intermediate_data['game_year']
        ab_number = item_intermediate_data['ab_number']
        inning_no = item_intermediate_data['inning_no']
        pitcher_id = item_intermediate_data['pitcher_id']
        batter_id = item_intermediate_data['batter_id']

        stadium_id = item_intermediate_data['stadium_id']
        state_delta_ids = item_intermediate_data['state_delta_ids']
        pitch_types = item_intermediate_data['pitch_types']
        pitcher_pos_id = item_intermediate_data['pitcher_pos_id']
        batter_pos_id = item_intermediate_data['batter_pos_id']
        rv_pitch_data = item_intermediate_data['rv_pitch_data']
        plate_x = item_intermediate_data['plate_x']
        plate_z = item_intermediate_data['plate_z']

        intermediate_pitcher_inputs = item_intermediate_data['raw_pitcher_inputs']
        pitcher_handedness_str = item_intermediate_data['raw_pitcher_inputs']['handedness'].upper()
        pitcher_handedness_id = self.handedness_id_map[pitcher_handedness_str]
        intermediate_batter_inputs = item_intermediate_data['raw_batter_inputs']
        batter_handedness_str = item_intermediate_data['raw_batter_inputs']['handedness'].upper()
        batter_handedness_id = self.handedness_id_map[batter_handedness_str]
        intermediate_matchup_inputs = item_intermediate_data['raw_matchup_inputs']
        raw_pitcher_inputs = []
        raw_batter_inputs = []
        raw_matchup_inputs = []

        for data_scope in self.batter_data_scopes_to_use:
            raw_batter_inputs.extend(intermediate_batter_inputs[data_scope])

        for data_scope in self.pitcher_data_scopes_to_use:
            raw_pitcher_inputs.extend(intermediate_pitcher_inputs[data_scope])

        for data_scope in self.matchup_data_scopes_to_use:
            raw_matchup_inputs.extend(intermediate_matchup_inputs[data_scope])

        pitch_types = [pt if pt is not None else 'NONE' for pt in pitch_types]
        pitch_types = [self.pitch_type_mapping[pt] for pt in pitch_types]
        pitcher_data = [raw_pitcher_inputs for _ in state_delta_ids]
        pitcher_id = [pitcher_id for _ in state_delta_ids]
        batter_data = [raw_batter_inputs for _ in state_delta_ids]
        batter_id = [batter_id for _ in state_delta_ids]
        matchup_data = [raw_matchup_inputs for _ in state_delta_ids]
        pitcher_pos_ids = [pitcher_pos_id for _ in state_delta_ids]
        batter_pos_ids = [batter_pos_id for _ in state_delta_ids]
        ab_number_ids = [ab_number for _ in state_delta_ids]
        inning = [inning_no for _ in state_delta_ids]
        stadium_id = [stadium_id for _ in state_delta_ids]
        pitcher_handedness_id = [pitcher_handedness_id for _ in state_delta_ids]
        batter_handedness_id = [batter_handedness_id for _ in state_delta_ids]

        if self.parse_plate_pos_to_id:
            plate_x = [int(px) if px is not None and px > 0 else 0 for px in
                       bucketize(plate_x, min=-2, max=2, n_buckets=8)]
            plate_z = [int(pz) if pz is not None and pz > 0 else 0 for pz in
                       bucketize(plate_z, min=0, max=4, n_buckets=9)]
        else:
            plate_x = [faux_normalize(px, -10.0, 10.0) if px is not None else 0.0 for px in plate_x]
            plate_z = [faux_normalize(pz, -6.0, 15.0) if pz is not None else 0.0 for pz in plate_z]

        pitch_types = torch.tensor(pitch_types)
        plate_x = torch.tensor(plate_x, dtype=torch.long if self.parse_plate_pos_to_id else torch.float)
        plate_z = torch.tensor(plate_z, dtype=torch.long if self.parse_plate_pos_to_id else torch.float)
        pitcher_data = torch.tensor(pitcher_data)
        pitcher_id = torch.tensor(pitcher_id)
        batter_data = torch.tensor(batter_data)
        batter_id = torch.tensor(batter_id)
        matchup_data = torch.tensor(matchup_data)
        pitcher_pos_ids = torch.tensor(pitcher_pos_ids)
        batter_pos_ids = torch.tensor(batter_pos_ids)
        ab_number = torch.tensor(ab_number_ids)
        inning_ids = torch.tensor(inning)
        inning = torch.tensor(inning)
        stadium_id = torch.tensor(stadium_id)
        pitcher_handedness_id = torch.tensor(pitcher_handedness_id)
        batter_handedness_id = torch.tensor(batter_handedness_id)
        ab_number_ids = torch.tensor(ab_number_ids)
        rv_pitch_data = torch.tensor(rv_pitch_data, dtype=torch.float)
        state_delta_ids = torch.tensor(state_delta_ids)
        ab_lengths = state_delta_ids.shape[0]
        game_pk = torch.tensor([game_pk])
        game_year = torch.tensor([game_year])

        item_data = {
            'game_pk': game_pk, 'game_year': game_year, 'inning': inning, 'inning_ids': inning_ids,
            'stadium_ids': stadium_id, 'pitcher_handedness_ids': pitcher_handedness_id,
            'batter_handedness_ids': batter_handedness_id, 'ab_number': ab_number, 'ab_number_ids': ab_number_ids,
            'state_delta_ids': state_delta_ids, 'ab_lengths': ab_lengths, 'pitch_types': pitch_types,
            'plate_x': plate_x, 'plate_z': plate_z, 'pitcher_inputs': pitcher_data, 'pitcher_id': pitcher_id,
            'batter_inputs': batter_data, 'batter_id': batter_id, 'matchup_inputs': matchup_data,
            'pitcher_pos_ids': pitcher_pos_ids, 'batter_pos_ids': batter_pos_ids, 'rv_pitch_data': rv_pitch_data,
        }

        return item_data

    def parse_item_intermediate(self, item_fp):
        """
        Parse all data in a given at-bat JSON file
        :param item_fp: filepath of the JSON at-bat file to parse
        :return: parsed at-bat JSON record
        """
        item_j = json.load(open(os.path.join(self.ab_data_dir, item_fp)))

        game_pk = item_j['game']['game_pk']
        game_year = item_j['game']['game_year']
        ab_number = item_j['game']['at_bat_number']
        inning_no = item_j['game']['inning']
        inning_topbot = item_j['game']['inning_topbot']
        outs_when_up = item_j['game']['outs_when_up']
        next_game_state = item_j['game']['next_game_state']
        pitcher_j = item_j['pitcher']
        pitcher_id = item_j['pitcher']['__id__']
        batter_j = item_j['batter']
        batter_id = item_j['batter']['__id__']
        matchup_j = item_j['matchup']

        home_team_str = item_j['game']['home_team']
        stadium_id = int(self.team_stadium_map[home_team_str][str(game_year)]) - 1

        pitcher_pos_str = self.player_bio_info_mapping[item_j['pitcher']['__id__']][self.player_pos_key]
        batter_pos_str = self.player_bio_info_mapping[item_j['batter']['__id__']][self.player_pos_key]
        if str(pitcher_pos_str) == 'nan':
            pitcher_pos_str = 'UNK'
        if str(batter_pos_str) == 'nan':
            batter_pos_str = 'UNK'

        on_1b = 0 if item_j['game']['on_1b'] in [None, 0] else 1
        on_2b = 0 if item_j['game']['on_2b'] in [None, 0] else 1
        on_3b = 0 if item_j['game']['on_3b'] in [None, 0] else 1
        next_game_state['on_1b'] = 0 if next_game_state['on_1b'] is None else 1
        next_game_state['on_2b'] = 0 if next_game_state['on_2b'] is None else 1
        next_game_state['on_3b'] = 0 if next_game_state['on_3b'] is None else 1
        pitcher_pos_id = self.player_pos_id_map[pitcher_pos_str]
        batter_pos_id = self.player_pos_id_map[batter_pos_str]
        pitcher_score = item_j['game']['fld_score']
        batter_score = item_j['game']['bat_score']

        n_balls, n_strikes, pitch_types, plate_x, plate_z, rv_pitch_data = parse_pitches(item_j['pitches'],
                                                                                         self.parse_plate_pos_to_id,
                                                                                         norm_vals=
                                                                                         self.record_norm_values[
                                                                                             'pitches'])
        last_gamestate = {'balls': 0, 'strikes': 0, 'outs': outs_when_up, 'score': batter_score,
                          'on_1b': on_1b, 'on_2b': on_2b, 'on_3b': on_3b}

        state_deltas, _ = find_state_deltas_v2(n_balls, n_strikes, outs_when_up, inning_topbot,
                                               next_game_state, batter_score, on_1b, on_2b, on_3b,
                                               last_state=last_gamestate)

        state_deltas = state_deltas[1:]
        state_delta_ids = [self.gamestate_vocab.get_id(sd) for sd in state_deltas]

        if -1 in state_delta_ids:
            print('*' * 50)
            print(os.path.join(self.ab_data_dir, item_fp))
            print(state_delta_ids)
            print('*' * 50)

        raw_pitcher_inputs = parse_json(pitcher_j, j_type='pitcher', max_value_data=self.record_norm_values)
        raw_batter_inputs = parse_json(batter_j, j_type='batter', max_value_data=self.record_norm_values)
        raw_matchup_inputs = parse_json(matchup_j, j_type='matchup', max_value_data=self.record_norm_values)

        intermediate_data = {
            'game_pk': game_pk, 'game_year': game_year, 'ab_number': ab_number, 'inning_no': inning_no,
            'pitcher_id': pitcher_id, 'batter_id': batter_id, 'stadium_id': stadium_id,
            'pitcher_pos_id': pitcher_pos_id, 'batter_pos_id': batter_pos_id, 'state_delta_ids': state_delta_ids,
            'raw_pitcher_inputs': raw_pitcher_inputs, 'raw_batter_inputs': raw_batter_inputs,
            'raw_matchup_inputs': raw_matchup_inputs, 'pitch_types': pitch_types, 'rv_pitch_data': rv_pitch_data,
            'plate_x': plate_x, 'plate_z': plate_z,
        }

        return intermediate_data
