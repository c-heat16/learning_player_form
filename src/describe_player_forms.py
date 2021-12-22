__author__ = 'Connor Heaton'

import os
import json
import time
import torch
import argparse

import numpy as np

from queue import Queue
from threading import Thread
from datetime import datetime
from argparse import Namespace
from models import PlayerFormModel
from datasets import PlayerFormDataset
from vocabularies import GeneralGamestateDeltaVocab

from batting_order_generator import BattingOrderGenerator


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


def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("huh?")


def autoconvert(s):
    if s in ['[BOS]', '[EOS]']:
        return s
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass

    if s[0] == '[' and s[-1] == ']':
        s = s[1:-1]
        s = [ss.strip().strip('\'') for ss in s.split(',')]

    return s


def read_model_args(fp):
    m_args = {}

    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '':
                arg, val = line.split('=')
                arg = arg.strip()
                val = val.strip()

                val = autoconvert(val)
                m_args[arg] = val

    m_args = Namespace(**m_args)
    return m_args


def read_player_career_data(data_dir, bad_data_fps, player_type='batter'):
    all_career_data = {}
    player_type_data_dir = os.path.join(data_dir, player_type)

    for player_filename in os.listdir(player_type_data_dir):
        player_id = int(player_filename[:-4])
        player_fp = os.path.join(player_type_data_dir, player_filename)
        player_career_data = {'game_ids': [], 'game_ab_fps': []}
        curr_game_pk = None

        with open(player_fp, 'r') as f:
            for line in f:
                line = line.strip()
                if not line == '':
                    if line in bad_data_fps:
                        print('* Bad file found - {} *'.format(line))
                    else:
                        ab_season, ab_filename = line.split('/')
                        ab_game_pk = int(ab_filename.split('-')[0])

                        if ab_game_pk != curr_game_pk:
                            player_career_data['game_ids'].append(ab_game_pk)
                            player_career_data['game_ab_fps'].append([])

                        player_career_data['game_ab_fps'][-1].append(line)
                        curr_game_pk = ab_game_pk

        all_career_data[player_id] = player_career_data

    return all_career_data


def load_model_and_dataset(model_ckpt_fp, model_args):
    print('Instantiating model...')
    model = PlayerFormModel(model_args).to('cuda:0')

    print('Loading model ckpt...')
    state_dict = torch.load(model_ckpt_fp, map_location={'cuda:0': 'cuda:0'})
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print('Creating vocab...')
    gamestate_vocab_bos_inning_no = getattr(model_args, 'gamestate_vocab_bos_inning_no', True)
    gamestate_vocab_bos_score_diff = getattr(model_args, 'gamestate_vocab_bos_score_diff', True)
    gamestate_vocab_use_balls_strikes = getattr(model_args, 'gamestate_vocab_use_balls_strikes', True)
    gamestate_vocab_use_base_occupancy = getattr(model_args, 'gamestate_vocab_use_base_occupancy', True)
    gamestate_vocab_use_score_diff = getattr(model_args, 'gamestate_vocab_use_score_diff', True)
    gamestate_vocab_use_outs = getattr(model_args, 'gamestate_vocab_use_outs', True)
    gamestate_n_innings = getattr(model_args, 'gamestate_n_innings', 10)
    gamestate_max_score_diff = getattr(model_args, 'gamestate_max_score_diff', 6)
    gamestate_vocab = GeneralGamestateDeltaVocab(bos_inning_no=gamestate_vocab_bos_inning_no,
                                                 max_inning_no=gamestate_n_innings,
                                                 bos_score_diff=gamestate_vocab_bos_score_diff,
                                                 bos_max_score_diff=gamestate_max_score_diff,
                                                 balls_delta=gamestate_vocab_use_balls_strikes,
                                                 strikes_delta=gamestate_vocab_use_balls_strikes,
                                                 outs_delta=gamestate_vocab_use_outs,
                                                 score_delta=gamestate_vocab_use_score_diff,
                                                 base_occ_delta=gamestate_vocab_use_base_occupancy,
                                                 )
    print('Creating dataset...')
    dataset = PlayerFormDataset(model_args, 'apply', gamestate_vocab)

    return model, dataset


def make_dataset_for_worker(model_args):
    gamestate_vocab_bos_inning_no = getattr(model_args, 'gamestate_vocab_bos_inning_no', True)
    gamestate_vocab_bos_score_diff = getattr(model_args, 'gamestate_vocab_bos_score_diff', True)
    gamestate_vocab_use_balls_strikes = getattr(model_args, 'gamestate_vocab_use_balls_strikes', True)
    gamestate_vocab_use_base_occupancy = getattr(model_args, 'gamestate_vocab_use_base_occupancy', True)
    gamestate_vocab_use_score_diff = getattr(model_args, 'gamestate_vocab_use_score_diff', True)
    gamestate_vocab_use_outs = getattr(model_args, 'gamestate_vocab_use_outs', True)
    gamestate_n_innings = getattr(model_args, 'gamestate_n_innings', 10)
    gamestate_max_score_diff = getattr(model_args, 'gamestate_max_score_diff', 6)
    gamestate_vocab = GeneralGamestateDeltaVocab(bos_inning_no=gamestate_vocab_bos_inning_no,
                                                 max_inning_no=gamestate_n_innings,
                                                 bos_score_diff=gamestate_vocab_bos_score_diff,
                                                 bos_max_score_diff=gamestate_max_score_diff,
                                                 balls_delta=gamestate_vocab_use_balls_strikes,
                                                 strikes_delta=gamestate_vocab_use_balls_strikes,
                                                 outs_delta=gamestate_vocab_use_outs,
                                                 score_delta=gamestate_vocab_use_score_diff,
                                                 base_occ_delta=gamestate_vocab_use_base_occupancy,
                                                 )

    dataset = PlayerFormDataset(model_args, 'apply', gamestate_vocab)
    dataset.do_cache = False
    dataset.do_mem_cache = True
    dataset.try_mt = False
    dataset.n_mt = 3
    dataset.memory_cache = {}

    return dataset


def describe_player_forms(model, model_args, player_career_data, out_dir, whole_game_record_dir, form_n_abs,
                          player_type, norm_vals, start_year, end_year, n_workers=8, cls_to_return='proj'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('out_dir: {}'.format(out_dir))
    # seasons = [2015, 2016, 2017, 2018, 2019]
    # seasons = [2017]
    seasons = [season for season in range(start_year, end_year + 1)]
    print('Seasons: {}'.format(', '.join([str(v) for v in seasons])))
    n_writes = 0
    n_games = 0
    dataset = None
    for season in seasons:
        print('Describing player forms from {} season...'.format(season))
        season_dir = os.path.join(whole_game_record_dir, 'by_season', str(season))
        season_out_dir = os.path.join(out_dir, str(season))
        if not os.path.exists(season_out_dir):
            os.makedirs(season_out_dir)

        out_fp_tmplt = os.path.join(season_out_dir, '{}-{}.npy')
        state_delta_fp_tmplt = os.path.join(season_out_dir, '{}-{}_recent_state_deltas.csv')
        pitch_event_fp_tmplt = os.path.join(season_out_dir, '{}-{}_recent_pitch_event_data.jsonl')

        print('Wiping dataset memory cache from last season...')
        del dataset
        dataset = make_dataset_for_worker(model_args)
        dataset.do_mem_cache = True

        generator_in_qs = [Queue() for _ in range(n_workers)]
        input_data_q = Queue()
        batting_order_generators = [BattingOrderGenerator(player_career_data,
                                                          dataset,
                                                          form_n_abs, i, in_q,
                                                          input_data_q, player_type) for i, in_q in
                                    enumerate(generator_in_qs)]
        generator_threads = [Thread(target=bog.work, args=()) for bog in batting_order_generators]

        print('Starting generator threads...')
        for gen_thread in generator_threads:
            gen_thread.start()

        print('Pushing game filepaths to gen qs...')
        n_total_games_this_season = 0
        for game_idx, game_filename in enumerate(os.listdir(season_dir)):
            n_total_games_this_season += 1
            game_pk = int(game_filename[:-5])
            game_fp = os.path.join(season_dir, game_filename)
            generator_in_qs[game_idx % n_workers].put([game_pk, game_fp])

        print('* n_total_games_this_season: {} *'.format(n_total_games_this_season))
        for q in generator_in_qs:
            q.put('[TERM]')

        n_term_rcvd = 0
        n_sleep = 0
        n_games_processed_this_season = 0
        start_time = time.time()
        while True:
            if input_data_q.empty():
                n_sleep += 1
                if n_sleep % 2 == 0:
                    print('Sleeping for input data...')
                time.sleep(15)
            else:
                in_data = input_data_q.get()
                if in_data == '[TERM]':
                    n_term_rcvd += 1
                    print('[{}] Orchestrator rcvd {} term signals...'.format(datetime.now().strftime("%H:%M:%S"),
                                                                             n_term_rcvd))

                    if n_term_rcvd == n_workers:
                        print('[{0}] RCVD {1} TERM SIGNALS'.format(datetime.now().strftime("%H:%M:%S"), n_term_rcvd))
                        break
                else:
                    game_pk, all_player_ids, agg_inputs = in_data
                    state_delta_ids = agg_inputs['state_delta_ids'].to('cuda:0', non_blocking=True)
                    pitcher_inputs = agg_inputs['pitcher_inputs'].to('cuda:0', non_blocking=True)
                    batter_inputs = agg_inputs['batter_inputs'].to('cuda:0', non_blocking=True)
                    matchup_inputs = agg_inputs['matchup_inputs'].to('cuda:0', non_blocking=True)
                    pitcher_pos_ids = agg_inputs['pitcher_pos_ids'].to('cuda:0', non_blocking=True)
                    batter_pos_ids = agg_inputs['batter_pos_ids'].to('cuda:0', non_blocking=True)
                    pitch_types = agg_inputs['pitch_types'].to('cuda:0', non_blocking=True)
                    plate_x = agg_inputs['plate_x'].to('cuda:0', non_blocking=True)
                    plate_z = agg_inputs['plate_z'].to('cuda:0', non_blocking=True)
                    rv_pitch_data = agg_inputs['rv_pitch_data'].to('cuda:0', non_blocking=True)
                    my_src_pad_mask = agg_inputs['my_src_pad_mask'].to('cuda:0', non_blocking=True)
                    model_src_pad_mask = agg_inputs['model_src_pad_mask'].to('cuda:0', non_blocking=True)
                    ab_lengths = agg_inputs['ab_lengths'].to('cuda:0', non_blocking=True)
                    record_ab_numbers = agg_inputs['record_ab_numbers'].to('cuda:0', non_blocking=True)
                    pitch_numbers = agg_inputs['pitch_numbers'].to('cuda:0', non_blocking=True)
                    stadium_ids = agg_inputs['stadium_ids'].to('cuda:0', non_blocking=True)

                    pitcher_handedness_ids = agg_inputs['pitcher_handedness_ids'].to('cuda:0', non_blocking=True)
                    batter_handedness_ids = agg_inputs['batter_handedness_ids'].to('cuda:0', non_blocking=True)

                    model_outputs = model.process_data(
                        gamestate_ids=state_delta_ids, pitcher_data=pitcher_inputs, batter_data=batter_inputs,
                        matchup_data=matchup_inputs, pitcher_pos_ids=pitcher_pos_ids,
                        batter_pos_ids=batter_pos_ids, gamestate_labels=None,
                        pitch_type_ids=pitch_types, plate_x_ids=plate_x, plate_z_ids=plate_z,
                        rv_pitch_data=rv_pitch_data,
                        model_src_pad_mask=model_src_pad_mask, ab_lengths=ab_lengths,
                        do_id_mask=False,
                        record_ab_numbers=record_ab_numbers,
                        pitch_numbers=pitch_numbers, stadium_ids=stadium_ids,
                        normalize_cls_proj=True, cls_to_return=cls_to_return,
                        pitcher_handedness=pitcher_handedness_ids,
                        batter_handedness=batter_handedness_ids,
                    )
                    cls_projections = model_outputs[0].squeeze().to('cpu').numpy()
                    state_delta_ids = state_delta_ids.to('cpu').numpy()
                    my_src_pad_mask = my_src_pad_mask.to('cpu').numpy()
                    rv_pitch_data = rv_pitch_data.to('cpu').numpy()
                    for player_idx in range(cls_projections.shape[0]):
                        player_cls_proj = cls_projections[player_idx]
                        player_id = all_player_ids[player_idx]
                        player_out_fp = out_fp_tmplt.format(game_pk, player_id)
                        np.save(player_out_fp, player_cls_proj)
                        n_writes += 1

                        player_state_deltas = state_delta_ids[player_idx].squeeze()
                        player_pad_mask = my_src_pad_mask[player_idx].squeeze()
                        player_pitch_data = rv_pitch_data[player_idx].squeeze()
                        deltas_for_file = []
                        recent_pitches = []

                        for t_idx in range(player_state_deltas.shape[0]):
                            if player_pad_mask[t_idx] == 1:
                                deltas_for_file.append(player_state_deltas[t_idx])
                                pitch_info = {
                                    'release_speed': float(player_pitch_data[t_idx, 0]),
                                    'release_pos_x': float(player_pitch_data[t_idx, 1]),
                                    'release_pos_y': float(player_pitch_data[t_idx, 2]),
                                    'release_pos_z': float(player_pitch_data[t_idx, 3]),
                                    'release_spin_rate': float(player_pitch_data[t_idx, 4]),
                                    'release_extension': float(player_pitch_data[t_idx, 5]),
                                    'hc_x': float(player_pitch_data[t_idx, 6]),
                                    'hc_y': float(player_pitch_data[t_idx, 7]),
                                    'vx0': float(player_pitch_data[t_idx, 8]),
                                    'vy0': float(player_pitch_data[t_idx, 9]),
                                    'vz0': float(player_pitch_data[t_idx, 10]),
                                    'ax': float(player_pitch_data[t_idx, 11]),
                                    'ay': float(player_pitch_data[t_idx, 12]),
                                    'az': float(player_pitch_data[t_idx, 13]),
                                    'hit_distance_sc': float(player_pitch_data[t_idx, 14]),
                                    'launch_speed': float(player_pitch_data[t_idx, 15]),
                                    'launch_angle': float(player_pitch_data[t_idx, 16]),
                                }
                                pitch_info = {k: v * norm_vals['pitches'][k] for k, v in pitch_info.items()}
                                recent_pitches.append(pitch_info)

                        deltas_for_file = ','.join([str(v) for v in deltas_for_file])
                        delta_out_file = state_delta_fp_tmplt.format(game_pk, player_id)
                        pitch_out_file = pitch_event_fp_tmplt.format(game_pk, player_id)
                        with open(delta_out_file, 'w+') as f:
                            f.write(deltas_for_file)

                        with open(pitch_out_file, 'w+') as f:
                            f.write('\n'.join([json.dumps(rp) for rp in recent_pitches]))

                    n_games += 1
                    n_games_processed_this_season += 1
                    if n_games_processed_this_season % 100 == 0:
                        elapsed_time = time.time() - start_time
                        t_per_g = elapsed_time / n_games_processed_this_season
                        print('[{0}] Orchestrator processed {1}/{2} games ({3:.2f}s/game)'.format(
                            datetime.now().strftime("%H:%M:%S"),
                            n_games_processed_this_season, n_total_games_this_season, t_per_g
                        ))

    print('n_writes: {}'.format(n_writes))
    print('n_games: {}'.format(n_games))
    print('done :)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ab_data', default='/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1',
                        help='Dir where data can be found')
    parser.add_argument('--career_data',
                        default='/home/czh/sata1/learning_player_form/player_career_data',
                        help='Where to find player career data')
    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records',
                        help='Where to find player career data')
    parser.add_argument('--start_year', default=2015, type=int)
    parser.add_argument('--end_year', default=2019, type=int)

    parser.add_argument('--model_ckpt',
                        default='../out/player_form_v2/20211211-193143/models/model_120e.pt',
                        )
    parser.add_argument('--out', default='/home/czh/sata1/learning_player_form/')
    parser.add_argument('--out_dir_tmplt', default='{}_form_v58')

    parser.add_argument('--bad_data_fps', default=['2015/414020-81.json', '2015/414264-19.json', '2015/415933-70.json',
                                                   '2016/448381-44.json', '2016/447752-28.json', '2017/492354-17.json',
                                                   '2017/492054-24.json', '2018/529755-56.json', '2018/529812-66.json',
                                                   '2019/567172-14.json',
                                                   ], type=str, nargs='+',
                        help='FPs w/ corrupted data (likely b/c statcast)')
    parser.add_argument('--arg_out_file', default='args.txt', help='File to write cli args to')
    # parser.add_argument('--player_id_to_n_pitches_map_fp',
    #                     default='/home/czh/sata1/learning_player_form/whole_game_records/'
    #                             'game_event_splits/stats/player_n_apps.json')
    # parser.add_argument('--norm_vals_fp',
    #                     default='/home/czh/sata1/learning_player_form/whole_game_records/'
    #                             'game_event_splits/stats/max_values.json')
    parser.add_argument('--norm_vals_fp', default='../config/max_values.json')

    parser.add_argument('--cls_to_return', default='proj')
    parser.add_argument('--n_workers', default=-1, type=int)
    args = parser.parse_args()

    model_basedir = os.path.split(os.path.split(args.model_ckpt)[0])[0]
    model_args_fp = os.path.join(model_basedir, 'args.txt')
    norm_vals = json.load(open(args.norm_vals_fp))

    print('Reading model args...')
    model_args = read_model_args(model_args_fp)
    args.player_type = model_args.player_type
    args.form_n_abs = model_args.view_size
    model_args.ab_data = args.ab_data
    model_args.career_data = args.career_data
    model_args.whole_game_record_dir = args.whole_game_record_dir
    # model_args.player_id_to_n_pitches_map_fp = args.player_id_to_n_pitches_map_fp

    if args.n_workers == -1:
        if args.player_type == 'batter':
            args.n_workers = 10
        else:
            args.n_workers = 3
    print('*** n_workers = {} ***'.format(args.n_workers))

    args.out = os.path.join(args.out, args.out_dir_tmplt.format(args.player_type))
    print('args.out: {}'.format(args.out))
    if os.path.exists(args.out):
        input('args.out exists')
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    args.arg_out_file = os.path.join(args.out, args.arg_out_file)
    args_d = vars(args)
    with open(args.arg_out_file, 'w+') as f:
        for k, v in args_d.items():
            f.write('{} = {}\n'.format(k, v))

    print('Reading {} career data...'.format(args.player_type))
    all_player_career_data = read_player_career_data(args.career_data, args.bad_data_fps, player_type=args.player_type)
    print('\t# {}: {}'.format(args.player_type, len(all_player_career_data)))

    model, dataset = load_model_and_dataset(args.model_ckpt, model_args)
    dataset.do_cache = False
    dataset.do_mem_cache = True

    with torch.no_grad():
        describe_player_forms(model, model_args, all_player_career_data, args.out, args.whole_game_record_dir,
                              args.form_n_abs, args.player_type, norm_vals,
                              start_year=args.start_year, end_year=args.end_year,
                              n_workers=args.n_workers,
                              cls_to_return=args.cls_to_return)
