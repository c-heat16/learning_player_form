__author__ = 'Connor Heaton'


import os
import json
import pickle
import datetime
import argparse

import numpy as np

from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.svm import LinearSVC, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from itertools import product


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


def read_lines(fp):
    items = []
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if not line == '':
                items.append(line)

    return items


def make_dataset(item_fps, whole_game_record_dir, form_dir, game_meta_stats):
    # game_by_season_dir = os.path.join(whole_game_record_dir, 'by_season')
    game_by_season_dir = whole_game_record_dir
    raw_xs, raw_meta = [], []
    raw_victor_labels = []
    raw_top_scores, raw_top_h, raw_top_hr = [], [], []
    raw_bot_scores, raw_bot_h, raw_bot_hr = [], [], []

    for item_fp in item_fps:
        item_season, item_game_pk = map(int, os.path.split(item_fp[:-5]))
        game_form_fp = os.path.join(form_dir, '{}.npy'.format(item_game_pk))
        item_game_j_fp = os.path.join(game_by_season_dir, item_fp)
        item_game_j = json.load(open(item_game_j_fp))

        home_score = item_game_j['home_score']
        away_score = item_game_j['away_score']
        home_team = item_game_j['home_team']
        away_team = item_game_j['away_team']
        if home_score == away_score:
            input('item_fp: {}'.format(item_fp))

        supplemental_x = np.array(game_meta_stats[str(item_game_pk)])
        # print('supplemental_x: {}'.format(supplemental_x.shape))
        victor_label = 0 if away_score > home_score else 1
        item_top_score = away_score
        item_top_h = item_game_j['ptb_starters']['top']['h']
        item_top_hr = item_game_j['ptb_starters']['top']['hr']
        raw_top_scores.append(item_top_score)
        raw_top_h.append(item_top_h)
        raw_top_hr.append(item_top_hr)

        item_bot_score = home_score
        item_bot_h = item_game_j['ptb_starters']['bot']['h']
        item_bot_hr = item_game_j['ptb_starters']['bot']['hr']
        raw_bot_scores.append(item_bot_score)
        raw_bot_h.append(item_bot_h)
        raw_bot_hr.append(item_bot_hr)

        item_data = [np.load(game_form_fp)]

        item_data = np.concatenate(item_data, axis=0)
        raw_xs.append(item_data)
        raw_meta.append(supplemental_x)
        raw_victor_labels.append(victor_label)

    xs = np.vstack(raw_xs)
    meta = np.vstack(raw_meta)
    victor_labels = np.array(raw_victor_labels)
    top_scores = np.array(raw_top_scores)
    top_h = np.array(raw_top_h)
    top_hr = np.array(raw_top_hr)
    bot_scores = np.array(raw_bot_scores)
    bot_h = np.array(raw_bot_h)
    bot_hr = np.array(raw_bot_hr)

    return xs, meta, victor_labels, top_scores, top_h, top_hr, bot_scores, bot_h, bot_hr


def rf_clf_parm_search(out_dir, args, train_x, train_y, test_x, test_y):
    out_fp = os.path.join(out_dir, 'model_scores.csv')
    models_dir = os.path.join(out_dir, 'models')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_fp_tmplt = os.path.join(models_dir, 'rf_{}n-est_{}max-feat_{}max-depth_{}min-split.pkl')

    parm_sets = product(
        args.rf_n_estimators,
        args.rf_max_features,
        args.rf_max_depth,
        args.rf_min_samples_split,
    )
    with open(out_fp, 'w+') as f:
        f.write('n_estimators,max_features,max_depth,min_samples_split,acc,f1\n')

    outdir = os.path.split(out_fp)[0]
    for n_estimators, max_features, max_depth, min_samples_split in parm_sets:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=16,
            n_jobs=24,
        )
        clf.fit(train_x, train_y)
        test_preds = clf.predict(test_x)
        acc = np.mean(test_preds == test_y)
        f1 = f1_score(test_y, test_preds)

        write_items = [n_estimators, max_features, max_depth, min_samples_split, acc, f1]
        write_line = ','.join([str(v) for v in write_items])
        print('\t{}'.format(write_line))
        with open(out_fp, 'a') as f:
            f.write('{}\n'.format(write_line))

        # model_save_fp = os.path.join(outdir, 'rf_{}est_{}maxfeat_{}maxdepth_{}minsplit.pkl'.format(n_estimators,
        #                                                                                            max_features,
        #                                                                                            max_depth,
        #                                                                                            min_samples_split))
        model_save_fp = model_fp_tmplt.format(n_estimators, str(max_features).replace('.', '-'), max_depth, min_samples_split)
        pickle.dump(clf, open(model_save_fp, 'wb'))


def logreg_clf_parm_search(out_fp, args, train_x, train_y, test_x, test_y):
    parm_sets = product(
        args.logreg_c,
        args.logreg_l1_ratio,
        args.logreg_solver,
        args.logreg_penalty,
    )
    with open(out_fp, 'w+') as f:
        f.write('C,l1_ratio,solver,penalty,acc,f1\n')

    for C, l1_ratio, solver, penalty in parm_sets:
        clf = LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty, l1_ratio=l1_ratio,
            random_state=16,
        )
        clf.fit(train_x, train_y)
        test_preds = clf.predict(test_x)
        acc = np.mean(test_preds == test_y)
        f1 = f1_score(test_y, test_preds)

        write_items = [C, l1_ratio, solver, penalty, acc, f1]
        write_line = ','.join([str(v) for v in write_items])
        with open(out_fp, 'a') as f:
            f.write('{}\n'.format(write_line))


def svm_clf_parm_search(out_fp, args, train_x, train_y, test_x, test_y):
    parm_sets = product(
        args.svm_c,
        args.svm_max_iter,
    )
    with open(out_fp, 'w+') as f:
        f.write('C,max_iter,acc,f1\n')

    for C, max_iter in parm_sets:
        clf = make_pipeline(StandardScaler(),
                            LinearSVC(C=C, max_iter=max_iter, random_state=16))
        clf.fit(train_x, train_y)
        test_preds = clf.predict(test_x)
        acc = np.mean(test_preds == test_y)
        f1 = f1_score(test_y, test_preds)

        write_items = [C, max_iter, acc, f1]
        write_line = ','.join([str(v) for v in write_items])
        with open(out_fp, 'a') as f:
            f.write('{}\n'.format(write_line))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole_game_record_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records')
    parser.add_argument('--splits_basedir',
                        default='/home/czh/sata1/learning_player_form/whole_game_splits')
    parser.add_argument('--form_dir',
                        default='/home/czh/sata1/learning_player_form/whole_game_records/by_season_from_form_v1')
    parser.add_argument('--form_subdir', default='pca-5')    # late_pca pca raw
    parser.add_argument('--game_meta_fp',
                        default='/home/czh/sata1/learning_player_form/game_meta_vectors_v1/game_meta_vectors.json')

    parser.add_argument('--out', default='../out/basic_parm_search', help='Directory to put output')

    parser.add_argument('--use_stats', default=False, type=str2bool)
    parser.add_argument('--stat_pca', default=False, type=str2bool)
    parser.add_argument('--use_form', default=False, type=str2bool)
    parser.add_argument('--use_meta', default=False, type=str2bool)
    parser.add_argument('--force_new_data', default=False, type=str2bool)

    parser.add_argument('--do_rf', default=False, type=str2bool)
    parser.add_argument('--do_logreg', default=False, type=str2bool)
    parser.add_argument('--do_svm', default=False, type=str2bool)
    parser.add_argument('--do_linreg', default=False, type=str2bool)

    # rf parms
    parser.add_argument('--rf_n_estimators',
                        default=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                        type=int, nargs='+')
    parser.add_argument('--rf_max_features', default=[-1, 0.25, 0.5], type=float, nargs='+')
    parser.add_argument('--rf_max_depth', default=[-1, 8, 10, 30, 50], type=int, nargs='+')
    parser.add_argument('--rf_min_samples_split', default=[2, 3, 4], type=int, nargs='+')

    # logreg parms
    parser.add_argument('--logreg_c', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], type=float, nargs='+')
    parser.add_argument('--logreg_l1_ratio', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                                                      0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                        type=float, nargs='+')
    parser.add_argument('--logreg_solver', default=['saga'], type=str, nargs='+')
    parser.add_argument('--logreg_penalty', default=['elasticnet'], type=str, nargs='+')

    # svm parms
    parser.add_argument('--svm_c', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], type=float, nargs='+')
    parser.add_argument('--svm_fit_intercept', default=[True, False], type=str2bool, nargs='+')
    parser.add_argument('--svm_max_iter', default=[1000, 5000, 10000], type=str2bool, nargs='+')

    # linreg parms
    parser.add_argument('--linreg_fit_intercept', default=[True, False], type=str2bool, nargs='+')
    parser.add_argument('--linreg_normalize', default=[True, False], type=str2bool, nargs='+')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    args.rf_max_features = [v if v > 0 else 'auto' for v in args.rf_max_features]
    args.rf_max_depth = [v if v > 0 else None for v in args.rf_max_depth]
    game_meta_stats = json.load(open(args.game_meta_fp))

    form_meta = '_'.join(os.path.split(args.form_dir)[-1].split('_')[-2:])

    train_whole_game_records_split_fp = os.path.join(args.splits_basedir, 'train.txt')
    test_whole_game_records_split_fp = os.path.join(args.splits_basedir, 'test.txt')
    victor_labels_train_y_fp = os.path.join(args.splits_basedir, 'victor_labels_train_y.npy')
    victor_labels_test_y_fp = os.path.join(args.splits_basedir, 'victor_labels_test_y.npy')

    form_train_x_fp = os.path.join(args.splits_basedir, 'train_x_{}.npy'.format(form_meta))
    form_test_x_fp = os.path.join(args.splits_basedir, 'test_x_{}.npy'.format(form_meta))
    meta_train_x_fp = os.path.join(args.splits_basedir, 'train_x_meta.npy')
    meta_test_x_fp = os.path.join(args.splits_basedir, 'test_x_meta.npy')

    home_stats_train_x_fp = os.path.join(args.splits_basedir, 'train_x_stats_home.npy')
    away_stats_train_x_fp = os.path.join(args.splits_basedir, 'train_x_stats_away.npy')
    matchup_stats_train_x_fp = os.path.join(args.splits_basedir, 'train_x_stats_matchup.npy')

    home_stats_test_x_fp = os.path.join(args.splits_basedir, 'test_x_stats_home.npy')
    away_stats_test_x_fp = os.path.join(args.splits_basedir, 'test_x_stats_away.npy')
    matchup_stats_test_x_fp = os.path.join(args.splits_basedir, 'test_x_stats_matchup.npy')

    all_train_x = []
    all_test_x = []
    data_desc = [form_meta]

    if args.use_form or args.use_meta:
        if not os.path.exists(form_train_x_fp) or args.force_new_data:
            print('Reading train items...')
            train_items = read_lines(train_whole_game_records_split_fp)
            print('\ttrain_items[:5]: {}'.format(train_items[:5]))

            print('Reading test items...')
            test_items = read_lines(test_whole_game_records_split_fp)
            print('\ttest_items[:5]: {}'.format(test_items[:5]))

            print('Reading train data...')
            train_data = make_dataset(train_items, args.whole_game_record_dir,
                                      os.path.join(args.form_dir, args.form_subdir), game_meta_stats)
            form_train_x = train_data[0]
            meta_train_x = train_data[1]
            victor_labels_train_y = train_data[2]

            print('Reading test data...')
            test_data = make_dataset(test_items, args.whole_game_record_dir,
                                     os.path.join(args.form_dir, args.form_subdir), game_meta_stats)
            form_test_x = test_data[0]
            meta_test_x = test_data[1]
            victor_labels_test_y = test_data[2]

            print('Saving arrays to files...')
            np.save(form_train_x_fp, form_train_x)
            np.save(meta_train_x_fp, meta_train_x)
            np.save(victor_labels_train_y_fp, victor_labels_train_y)

            np.save(form_test_x_fp, form_test_x)
            np.save(meta_test_x_fp, meta_test_x)
            np.save(victor_labels_test_y_fp, victor_labels_test_y)
        else:
            print('Reading arrays from files...')
            form_train_x = np.load(form_train_x_fp)
            form_test_x = np.load(form_test_x_fp)
            meta_train_x = np.load(meta_train_x_fp)
            meta_test_x = np.load(meta_test_x_fp)

        if args.use_form:
            print('*** Using Form ***')
            all_train_x.append(form_train_x)
            all_test_x.append(form_test_x)
            print('\tform_train_x: {}'.format(form_train_x.shape))
            print('\tform_test_x: {}'.format(form_test_x.shape))
            data_desc.append('form')

        if args.use_meta:
            print('*** Using Meta ***')
            print('\tmeta_train_x: {}'.format(meta_train_x.shape))
            print('\tmeta_test_x: {}'.format(meta_test_x.shape))
            all_train_x.append(meta_train_x)
            all_test_x.append(meta_test_x)
            data_desc.append('meta')

        if args.use_stats:
            print('*** Using Stats ***')
            home_stats_train_x = np.load(home_stats_train_x_fp)
            away_stats_train_x = np.load(away_stats_train_x_fp)
            matchup_stats_train_x = np.load(matchup_stats_train_x_fp)
            home_stats_test_x = np.load(home_stats_test_x_fp)
            away_stats_test_x = np.load(away_stats_test_x_fp)
            matchup_stats_test_x = np.load(matchup_stats_test_x_fp)

            stat_train_x = np.concatenate([home_stats_train_x, away_stats_train_x, matchup_stats_train_x], axis=-1)
            stat_test_x = np.concatenate([home_stats_test_x, away_stats_test_x, matchup_stats_test_x], axis=-1)

            print('\tstat_train_x: {}'.format(stat_train_x.shape))
            print('\tstat_test_x: {}'.format(stat_test_x.shape))
            all_train_x.append(stat_train_x)
            all_test_x.append(stat_test_x)
            data_desc.append('stats')

    train_x = np.concatenate(all_train_x, axis=-1)
    test_x = np.concatenate(all_test_x, axis=-1)
    print('train_x: {} test_x: {}'.format(train_x.shape, test_x.shape))

    train_y = np.load(victor_labels_train_y_fp)
    test_y = np.load(victor_labels_test_y_fp)
    print('train_y: {} test_y: {}'.format(train_y.shape, test_y.shape))
    u_test_y, cnt_test_y = np.unique(test_y, return_counts=True)
    comb_y_meta = sorted(zip(u_test_y, cnt_test_y), key=lambda x: x[0], reverse=False)
    for u_y, cnt_y in comb_y_meta:
        print('u_y: {}\tcnt_y: {}'.format(u_y, cnt_y))

    data_desc = '-'.join(data_desc)
    print('data_desc: {}'.format(data_desc))
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.do_rf:
        out_dir = os.path.join(args.out, 'rf_clf_parm_search_{}_{}'.format(data_desc, curr_time))
        # out_fp = os.path.join(out_dir, 'model_scores.csv')
        print('out_fp: {}'.format(out_dir))
        rf_clf_parm_search(out_dir, args, train_x, train_y, test_x, test_y)
    elif args.do_logreg:
        out_fp = os.path.join(args.out, 'logreg_clf_parm_search_{}_{}.csv'.format(data_desc, curr_time))
        print('out_fp: {}'.format(out_fp))
        logreg_clf_parm_search(out_fp, args, train_x, train_y, test_x, test_y)
    elif args.do_svm:
        out_fp = os.path.join(args.out, 'svm_clf_parm_search_{}_{}.csv'.format(data_desc, curr_time))
        print('out_fp: {}'.format(out_fp))
        svm_clf_parm_search(out_fp, args, train_x, train_y, test_x, test_y)





