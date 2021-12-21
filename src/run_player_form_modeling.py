__author__ = 'Connor Heaton'

import os
import torch
import datetime
import argparse

import numpy as np
import torch.multiprocessing as mp

from runners import PlayerFormRunner


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
        s = [ss.strip('\'') for ss in s.split(',')]

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

    return m_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab_data', default='/home/czh/sata1/learning_player_form/ab_seqs/ab_seqs_v1',
                        help='Dir where AB data can be found')
    parser.add_argument('--career_data',
                        default='/home/czh/sata1/learning_player_form/player_career_data',
                        help='Where to find player career data')
    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records',
                        help='Where to find player career data')
    parser.add_argument('--record_norm_values_fp',
                        default='../config/max_values.json')

    parser.add_argument('--out', default='../out/player_form', help='Directory to put output')
    parser.add_argument('--config_dir', default='../config', help='Directory to find config data')
    parser.add_argument('--player_type', default='batter', help='Type of player to model. batter or pitcher')
    parser.add_argument('--do_mem_cache', default=False, type=str2bool,
                        help='Boolean if AB records should be saved to disc as tensors to be reused later')

    # general modeling parms
    parser.add_argument('--train', default=True, type=str2bool)
    parser.add_argument('--dev', default=True, type=str2bool)
    parser.add_argument('--test', default=False, type=str2bool)
    parser.add_argument('--epochs', default=5000, type=int, help='# epochs to train for')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size to use')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--l2', default=0.01, type=float)
    parser.add_argument('--general_dropout_prob', default=0.15, type=float)
    parser.add_argument('--token_mask_pct', default=0.15, type=float)
    parser.add_argument('--mask_override_prob', default=0.15, type=float)
    parser.add_argument('--n_grad_accum', default=1, type=int)
    parser.add_argument('--n_warmup_iters', default=1500, type=int)
    parser.add_argument('--mgsm_weight', default=1.0, type=float)
    parser.add_argument('--con_weight', default=1.0, type=float)

    parser.add_argument('--distribution_based_player_sampling_prob', default=0.15, type=float)
    parser.add_argument('--full_ab_window_size_prob', default=1.0, type=float)

    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--n_attn', default=8, type=int)
    parser.add_argument('--n_proj_layers', default=2, type=int)
    parser.add_argument('--n_raw_data_proj_layers', default=2, type=int)
    parser.add_argument('--complete_embd_dim', default=512, type=int)
    parser.add_argument('--gamestate_embd_dim', default=256, type=int)
    parser.add_argument('--raw_pitcher_data_dim', default=552, type=int)  # all scopes = 552, last_15|this_game = 274
    parser.add_argument('--raw_batter_data_dim', default=578, type=int)  # all scopes = 578, last_15|this_game = 274
    parser.add_argument('--raw_matchup_data_dim', default=411, type=int)  # all scopes = 411, this_game = 137
    parser.add_argument('--pitch_type_embd_dim', default=5, type=int)
    parser.add_argument('--player_pos_embd_dim', default=5, type=int)
    parser.add_argument('--n_ab_pitch_no_embds', default=25, type=int)

    # data / modeling parms
    parser.add_argument('--loss_objective', default='supcon')
    parser.add_argument('--proj_dim', default=64, type=int)

    parser.add_argument('--first_principles', default=False, type=str2bool)
    parser.add_argument('--use_statcast_data', default=True, type=str2bool)
    parser.add_argument('--use_player_data', default=True, type=str2bool)

    parser.add_argument('--both_player_positions', default=False, type=str2bool)
    parser.add_argument('--use_handedness', default=True, type=str2bool)
    parser.add_argument('--both_player_handedness', default=False, type=str2bool)

    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--min_view_step_size', default=5, type=int)
    parser.add_argument('--max_view_step_size', default=5, type=int)
    parser.add_argument('--form_ab_window_size', default=20, type=int, help='# of ABs to include in representation')
    parser.add_argument('--min_ab_to_be_included_in_dataset', default=60, type=int)
    parser.add_argument('--min_form_ab_window_size', default=20, type=int, help='# of ABs to include in representation')
    parser.add_argument('--max_seq_len', default=130, type=int)
    parser.add_argument('--max_view_len', default=105, type=int)
    parser.add_argument('--view_size', default=15, type=int)
    parser.add_argument('--n_plate_x_ids', default=8, type=int)
    parser.add_argument('--n_plate_z_ids', default=9, type=int)
    parser.add_argument('--parse_plate_pos_to_id', default=False, type=str2bool)
    parser.add_argument('--plate_pos_embd_dim', default=5, type=int)
    parser.add_argument('--n_stadium_ids', default=35, type=int)
    parser.add_argument('--stadium_embd_dim', default=5, type=int)
    parser.add_argument('--handedness_embd_dim', default=5, type=int)

    parser.add_argument('--player_pos_source', default='mlb')
    parser.add_argument('--pitch_type_config_fp', default='../config/pitch_type_id_mapping.json')
    parser.add_argument('--player_bio_info_fp', default='../config/statcast_id_to_bio_info.json')
    parser.add_argument('--player_id_map_fp', default='../config/all_player_id_mapping.json')
    parser.add_argument('--team_stadiums_fp', default='../config/team_stadiums.json')

    # vocab parms
    parser.add_argument('--gamestate_vocab_bos_inning_no', default=True, type=str2bool)
    parser.add_argument('--gamestate_vocab_bos_score_diff', default=True, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_balls_strikes', default=True, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_base_occupancy', default=True, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_inning_no', default=False, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_inning_topbot', default=False, type=str2bool)
    parser.add_argument('--gamestate_vocab_use_score_diff', default=True, type=str2bool)
    parser.add_argument('--gamestate_n_innings', default=10, type=int)
    parser.add_argument('--gamestate_max_score_diff', default=6, type=int)
    parser.add_argument('--gamestate_vocab_use_outs', default=True, type=str2bool)

    # logging parms
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--ckpt_file', default=None)
    parser.add_argument('--ckpt_file_tmplt', default='model_{}e.pt')
    parser.add_argument('--warm_start', default=False, type=str2bool)
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--log_every', default=10, type=int)
    parser.add_argument('--save_model_every', default=10, type=int)
    parser.add_argument('--save_preds_every', default=1, type=int)
    parser.add_argument('--summary_every', default=25, type=int)
    parser.add_argument('--dev_every', default=1, type=int)
    parser.add_argument('--arg_out_file', default='args.txt', help='File to write cli args to')
    parser.add_argument('--verbosity', default=0, type=int)
    parser.add_argument('--grad_summary', default=True, type=str2bool)
    parser.add_argument('--grad_summary_every', default=100, type=int)
    parser.add_argument('--bad_data_fps', default=['2015/414020-81.json', '2015/414264-19.json', '2015/415933-70.json',
                                                   '2016/448381-44.json', '2016/447752-28.json', '2017/492354-17.json',
                                                   '2017/492054-24.json', '2018/529755-56.json', '2018/529812-66.json',
                                                   '2019/567172-14.json'], type=str, nargs='+',
                        help='FPs w/ corrupted data (b/c statcast)')

    # hardware parms
    parser.add_argument('--gpus', default=[0], help='Which GPUs to use', type=int, nargs='+')
    parser.add_argument('--port', default='12345', help='Port to use for DDP')
    parser.add_argument('--on_cpu', default=False, type=str2bool)
    parser.add_argument('--n_data_workers', default=4, help='# threads used to fetch data *PER DEVICE/GPU*', type=int)

    parser.add_argument('--batter_data_scopes_to_use',
                        default=['career', 'season', 'last15', 'this_game'],
                        type=str, nargs='+', )
    parser.add_argument('--pitcher_data_scopes_to_use',
                        default=['career', 'season', 'last15', 'this_game'],
                        type=str, nargs='+', )
    parser.add_argument('--matchup_data_scopes_to_use',
                        default=['career', 'season', 'this_game'],
                        type=str, nargs='+', )
    args = parser.parse_args()
    args.world_size = len(args.gpus)

    print('args:\n{}'.format(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)

    run_modes = []
    if args.train:
        run_modes.append('train')

        # directories should only be made if training new model
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print('*' * len('* Model time ID: {} *'.format(curr_time)))
        print('* Model time ID: {} *'.format(curr_time))
        print('*' * len('* Model time ID: {} *'.format(curr_time)))

        args.out = os.path.join(args.out, curr_time)
        os.makedirs(args.out)

        args.tb_dir = os.path.join(args.out, 'tb_dir')
        os.makedirs(args.tb_dir)

        args.model_save_dir = os.path.join(args.out, 'models')
        os.makedirs(args.model_save_dir)

        args.model_log_dir = os.path.join(args.out, 'logs')
        os.makedirs(args.model_log_dir)

        args.arg_out_file = os.path.join(args.out, args.arg_out_file)
        args_d = vars(args)
        with open(args.arg_out_file, 'w+') as f:
            for k, v in args_d.items():
                f.write('{} = {}\n'.format(k, v))
    if args.dev:
        run_modes.append('dev')
    if args.test:
        run_modes.append('test')

    if (args.dev or args.test) and not args.train:
        args.out = os.path.dirname(args.ckpt_file)
        if os.path.basename(args.out) == 'models':
            args.out = os.path.dirname(args.out)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    for mode in run_modes:
        print('Creating {} distributed models for {}...'.format(len(args.gpus), mode))
        mp.spawn(PlayerFormRunner, nprocs=len(args.gpus), args=(mode, args))

    print('Finished!')




