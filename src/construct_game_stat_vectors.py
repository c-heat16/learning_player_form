__author__ = 'Hayden Long and Connor Heaton'

import numpy as np
import argparse
import json
import os


def strip_data(fp, desired_attrs=None, max_values=None, attr_name_fp=None):
    j = json.load(open(fp))
    storage_dict = {}

    home_starter = int(j['home_starter']['__id__'])
    home_batting_order = j['home_batting_order']
    away_starter = int(j['away_starter']['__id__'])
    away_batting_order = j['away_batting_order']

    home_batter_names_of_attributes = []
    away_batter_names_of_attributes = []
    away_matchup_names_of_attributes = []
    home_matchup_names_of_attributes = []
    pitcher_matchup_names_of_attributes = []

    for layer1_keys, layer1_values in desired_attrs.items():
        my_curr_layer1_keys = storage_dict.get(layer1_keys, {})

        if 'batter' in layer1_keys:
            effective_keys = layer1_keys

            for player_id, player_v in j[layer1_keys].items():

                for layer2_keys, layer2_values in layer1_values.items():

                    my_curr_layer2_keys = my_curr_layer1_keys.get(
                        layer2_keys, {})

                    for last_layer_attribute in layer2_values:
                        my_curr_last_layer_attributes = my_curr_layer2_keys.get(
                            last_layer_attribute, [])
                        my_curr_last_layer_attributes.append(
                            player_v[layer2_keys][last_layer_attribute] / max_values['batter'][layer2_keys][last_layer_attribute])

                        if layer2_keys not in home_batter_names_of_attributes:
                            if 'home_batter' in layer1_keys:
                                home_batter_names_of_attributes.append(
                                    layer1_keys + "_" + layer2_keys + "_" + last_layer_attribute)

                        if layer2_keys not in away_batter_names_of_attributes:
                            if 'away_batter' in layer1_keys:
                                away_batter_names_of_attributes.append(
                                    layer1_keys + "_" + layer2_keys + "_" + last_layer_attribute)

                        my_curr_layer2_keys[last_layer_attribute] = my_curr_last_layer_attributes

                    my_curr_layer1_keys[layer2_keys] = my_curr_layer2_keys

        elif 'matchup_data' in layer1_keys:
            # pass
            for matchup, matchup_v in j[layer1_keys].items():
                pitch_id, bat_id = map(int, matchup.split('-'))

                if bat_id in home_batting_order and pitch_id == away_starter:
                    effective_keys = 'home_' + layer1_keys

                    for layer2_keys, layer2_values in layer1_values.items():
                        my_curr_layer2_keys = my_curr_layer1_keys.get(
                            layer2_keys, {})

                        if 'matchup' in layer2_keys:
                            layer2_keys = effective_keys
                            #del storage_dict['matchup']

                        elif 'away_matchup_data' in layer2_keys:
                            layer2_keys.append(effective_keys)

                        for layer3_keys, layer3_values in layer2_values.items():
                            my_curr_layer3_keys = my_curr_layer2_keys.get(
                                layer3_keys, {})

                            for last_layer_attribute in layer3_values:

                                my_curr_last_layer_attributes = my_curr_layer3_keys.get(
                                    last_layer_attribute, [])
                                my_curr_last_layer_attributes.append(
                                    matchup_v[layer3_keys][last_layer_attribute] / max_values['matchup'][layer3_keys][last_layer_attribute])

                                if last_layer_attribute not in home_matchup_names_of_attributes:
                                    home_matchup_names_of_attributes.append(
                                        layer1_keys + "_" + layer2_keys + "_" + last_layer_attribute)

                                my_curr_layer3_keys[last_layer_attribute] = my_curr_last_layer_attributes

                            my_curr_layer2_keys[layer3_keys] = my_curr_layer3_keys
                        my_curr_layer1_keys[layer2_keys] = my_curr_layer2_keys

                elif bat_id in away_batting_order and pitch_id == home_starter:
                    effective_keys = 'away_' + layer1_keys

                    for layer2_keys, layer2_values in layer1_values.items():
                        my_curr_layer2_keys = my_curr_layer1_keys.get(
                            layer2_keys, {})

                        if 'matchup' in layer2_keys:
                            layer2_keys = effective_keys
                            #del storage_dict['matchup']

                        elif 'home_matchup_data' in layer2_keys:
                            layer2_keys.append(effective_keys)

                        for layer3_keys, layer3_values in layer2_values.items():
                            my_curr_layer3_keys = my_curr_layer2_keys.get(
                                layer3_keys, {})
                            # input(layer3_keys)

                            for last_layer_attribute in layer3_values:

                                my_curr_last_layer_attributes = my_curr_layer3_keys.get(
                                    last_layer_attribute, [])

                                my_curr_last_layer_attributes.append(
                                    matchup_v[layer3_keys][last_layer_attribute] / max_values['matchup'][layer3_keys][last_layer_attribute])

                                if last_layer_attribute not in away_matchup_names_of_attributes:
                                    away_matchup_names_of_attributes.append(
                                        layer1_keys + "_" + layer2_keys + "_" + last_layer_attribute)

                                my_curr_layer3_keys[last_layer_attribute] = my_curr_last_layer_attributes

                            my_curr_layer2_keys[layer3_keys] = my_curr_layer3_keys
                        my_curr_layer1_keys[layer2_keys] = my_curr_layer2_keys

        else:
            for layer2_keys, layer2_values in layer1_values.items():
                my_curr_layer2_keys = my_curr_layer1_keys.get(layer2_keys, {})

                for last_layer_attribute in layer2_values:

                    my_curr_last_layer_attributes = my_curr_layer2_keys.get(
                        last_layer_attribute, [])

                    my_curr_last_layer_attributes.append(
                        j[layer1_keys][layer2_keys][last_layer_attribute] / max_values['pitcher'][layer2_keys][last_layer_attribute])

                    if last_layer_attribute not in pitcher_matchup_names_of_attributes:
                        pitcher_matchup_names_of_attributes.append(
                            layer1_keys + "_" + layer2_keys + "_" + last_layer_attribute)

                    my_curr_layer2_keys[last_layer_attribute] = my_curr_last_layer_attributes

                my_curr_layer1_keys[layer2_keys] = my_curr_layer2_keys

        storage_dict[layer1_keys] = my_curr_layer1_keys

        if('matchup_data') in layer1_keys:
            for layer2_keys in storage_dict[layer1_keys].keys():
                for layer3_keys in storage_dict[layer1_keys][layer2_keys].keys():
                    for last_layer_attribute in storage_dict[layer1_keys][layer2_keys][layer3_keys].keys():
                        if type(storage_dict[layer1_keys][layer2_keys][layer3_keys][last_layer_attribute]) is not float:
                            storage_dict[layer1_keys][layer2_keys][layer3_keys][last_layer_attribute] = sum(
                                storage_dict[layer1_keys][layer2_keys][layer3_keys][last_layer_attribute])/len(storage_dict[layer1_keys][layer2_keys][layer3_keys])
                        else:
                            pass
                        #print('last_layer_attribute: {}'.format(last_layer_attribute))
                        #input('len(temp): {}'.format(len(temp)))
                        # print(temp)

        else:
            for layer2_keys in storage_dict[layer1_keys].keys():
                for layer3_keys in storage_dict[layer1_keys][layer2_keys].keys():
                    # print(layer3_keys)
                    # input("else")
                    storage_dict[layer1_keys][layer2_keys][layer3_keys] = sum(
                        storage_dict[layer1_keys][layer2_keys][layer3_keys])/len(storage_dict[layer1_keys][layer2_keys][layer3_keys])
                   # print(storage_dict[layer1_keys][layer2_keys][layer3_keys])

    concat_attribute_names(home_batter_names_of_attributes, away_batter_names_of_attributes,
                           home_matchup_names_of_attributes, away_matchup_names_of_attributes,
                           pitcher_matchup_names_of_attributes, attr_name_fp)

    return storage_dict


def concat_attribute_names(home_batter, away_batter, home_matchup, away_matchup, pitcher, attr_name_fp):
    # attribute_array_fp = 'tryThis1.npy'

    # attribute_array = []
    # attribute_array2 = []
    # full_attribute_array = []

    if os.path.exists(attr_name_fp):

        pass

    else:
        attribute_array = []
        attribute_array.extend(list(set(home_batter)))
        attribute_array.extend(list(set(away_batter)))
        attribute_array.extend(list(set(home_matchup)))
        attribute_array.extend(list(set(away_matchup)))
        attribute_array.extend(list(set(pitcher)))

        with open(attr_name_fp, 'w+') as f:
            f.write('\n'.join(attribute_array))


def find_the_result(fp):

    j = json.load(open(fp))

    result = []

    winner = 0 if j['home_score'] > j['away_score'] else 1

    result.append(winner)

    return result


def create_data_from_split_file(split_fp, whole_game_records_dir, desired_attrs, max_values):
    features = []
    labels = []
    prev_shape = None
    split_basedir, _ = os.path.split(split_fp)
    attr_name_fp = os.path.join(split_basedir, 'attr_names.txt')

    with open(split_fp, 'r') as f:
        for myline in f:
            strippedline = myline.strip()

            if strippedline != '':
                fp = os.path.join(whole_game_records_dir, strippedline)
                # print(fp)

                if os.path.exists(fp):
                    my_metrics = strip_data(fp, desired_attrs, max_values, attr_name_fp)
                    result = find_the_result(fp)

                    out = []
                    for k, v in my_metrics.items():
                        if 'matchup_data' in k:
                            for kk, vv in v.items():
                                for kkk, vvv in vv.items():
                                    for kkkk, vvvv in vvv.items():
                                        out.append(vvvv)
                        else:
                            for kk, vv in v.items():
                                for kkk, vvv in vv.items():
                                    if type(vvv) is float:
                                        out.append(vvv)

                    out = np.array(out).reshape(-1)
                    features.append(out)
                    labels.append(result)

                    if out.shape != prev_shape:
                        prev_shape = out.shape

    features = np.vstack(features)
    labels = np.array(labels)

    return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records')
    parser.add_argument('--splits_basedir',
                        default='/home/czh/sata1/learning_player_form/whole_game_splits')

    args = parser.parse_args()

    agg_desired_attrs = {
        'home': {
            'home_starter': {
                'career': [
                    'seasons_played', 'career_opp_avg', 'career_strike_pct', 'career_ball_pct',
                    'career_zone_pct', 'career_swstr_pct', 'career_pull_pct', 'career_cent_pct', 'career_oppo_pct',
                    'career_soft_pct', 'career_med_pct', 'career_hard_pct', 'career_f_strike_pct', 'career_era',
                    'career_whip_avg', 'career_w_avg', 'career_swing_pct_avg', 'career_contact_pct_avg',
                    'career_war_avg'
                ],
                'season': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ],
                'last15': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ]
            },
            'home_batters': {
                'career': [
                    'season_played', 'career_strike_pct', 'career_ball_pct', 'career_zone_pct', 'career_swstr_pct',
                    'career_pull_pct', 'career_cent_pct', 'career_oppo_pct', 'career_soft_pct', 'career_med_pct',
                    'career_hard_pct', 'career_war_avg', 'career_obp_avg', 'career_slg_avg'
                ],
                'season': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ],
                'last15': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ]
            },
        },
        'away': {
            'away_starter': {
                'career': [
                    'seasons_played', 'career_opp_avg', 'career_strike_pct', 'career_ball_pct',
                    'career_zone_pct', 'career_swstr_pct', 'career_pull_pct', 'career_cent_pct', 'career_oppo_pct',
                    'career_soft_pct', 'career_med_pct', 'career_hard_pct', 'career_f_strike_pct', 'career_era',
                    'career_whip_avg', 'career_w_avg', 'career_swing_pct_avg', 'career_contact_pct_avg',
                    'career_war_avg'
                ],
                'season': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ],
                'last15': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ]
            },
            'away_batters': {
                'career': [
                    'season_played', 'career_strike_pct', 'career_ball_pct', 'career_zone_pct', 'career_swstr_pct',
                    'career_pull_pct', 'career_cent_pct', 'career_oppo_pct', 'career_soft_pct', 'career_med_pct',
                    'career_hard_pct', 'career_war_avg', 'career_obp_avg', 'career_slg_avg'
                ],
                'season': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ],
                'last15': [
                    'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                    'walk_pct',
                    'strike_pct', 'ball_pct'
                ]
            },
        },
        'matchup': {
            'matchup_data': {
                'matchup': {
                    'career': [
                        'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                        'walk_pct', 'strike_pct', 'ball_pct', 'strikeout_pct', 'avg_hist_distance', 'avg_launch_speed',
                        'avg_launch_angle'
                    ],
                    'season': [
                        'n_batters_faced', 'total_pitch_count', 'single_pct', 'double_pct', 'triple_pct', 'hr_pct',
                        'walk_pct', 'strike_pct', 'ball_pct', 'strikeout_pct', 'avg_hist_distance', 'avg_launch_speed',
                        'avg_launch_angle'
                    ]
                },
            }
        }
    }
    # game_representation = {
    #     'home_team', 'away_team', 'home_score', 'away_score', 'game_pk', 'home_batting_order', 'away_batting_order'
    # }
    max_values_fp = '../config/max_values.json'
    max_values = json.load(open(max_values_fp))
    games_summary_dir = args.whole_game_record_dir

    for attr_set_name, desired_attrs in agg_desired_attrs.items():
        print('Creating {} data...'.format(attr_set_name))
        for split_name in ['train', 'test', 'dev']:
            print('\tProcessing {} split...'.format(split_name))
            split_fp = os.path.join(args.splits_basedir, '{}.txt'.format(split_name))
            out_fp = os.path.join(args.splits_basedir, '{}_x_stats_{}.npy'.format(split_name, attr_set_name))
            print('\t\tsplit_fp: {}'.format(split_fp))
            print('\t\tout_fp: {}'.format(out_fp))
            features, _ = create_data_from_split_file(split_fp, games_summary_dir, desired_attrs, max_values)
            print('\t\tfeatures: {}'.format(features.shape))
            print('\t\tsaving...')
            np.save(out_fp, features)
