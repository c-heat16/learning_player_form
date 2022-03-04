__author__ = 'Connor Heaton'

import math
import pybaseball as pb
from pandas._libs.missing import NAType


def check_nan_value(x):
    nan = False
    try:
        nan = math.isnan(x)
    except Exception as ex:
        nan = type(x) == NAType

    return nan


def remove_dollar_sign(x):
    x = str(x)
    if x.startswith('$'):
        x = x[1:]
    elif x.startswith('($'):
        x = x[2:-1]

    return x


class PyBaseballWorker(object):
    def __init__(self, args, out_q, date_ranges, t_id):
        self.args = args
        self.out_q = out_q
        self.date_ranges = date_ranges
        self.t_id = t_id
        self.term_item = self.args.term_item
        self.summary_every = self.args.pb_summary_every
        self.n_items_processed = 0

        print('PyBaseball worker {} has {} dates in date range list...'.format(self.t_id, len(self.date_ranges)))

    def query_statcast(self):
        for range_idx, (q_start_date, q_end_date) in enumerate(self.date_ranges):
            range_data = pb.statcast(start_dt=q_start_date.strftime("%Y-%m-%d"),
                                     end_dt=q_end_date.strftime("%Y-%m-%d"),
                                     verbose=False)
            for item_data in range_data.values.tolist():
                # print('item_data:\n{}'.format(item_data))
                # print('Raw item data: {}'.format(item_data))
                item_data[1] = str(item_data[1].strftime("%Y-%m-%d"))
                item_data = [None if check_nan_value(id_) else id_ for id_ in item_data]
                # in a recent update, pybaseball returns 3 more values for statcast calls
                item_data = item_data[:-3]
                # print('len(item_data): {}'.format(len(item_data)))
                # item_data = (None if check_nan_value(id_) else id_ for id_ in item_data)
                # print('item_data: {}'.format(item_data))
                self.out_q.put(item_data)
                self.n_items_processed += 1

                if self.n_items_processed % self.summary_every == 0:
                    print('PyBaseballWorker {} pushed {} statcast items...'.format(self.t_id, self.n_items_processed))
                    print('\tcurr date range: {} to {}'.format(q_start_date.strftime("%Y-%m-%d"),
                                                               q_end_date.strftime("%Y-%m-%d")))
                    print('\tdate range {} of {}'.format(range_idx, len(self.date_ranges)))

        # print('* Worker {} pushed term signal for statcast *'.format(self.t_id))
        self.out_q.put(self.term_item)
        print('* PyBaseballWorker {} pushed term signal for statcast\n\ttotal items pushed: {}'.format(self.t_id,
                                                                                             self.n_items_processed))

    def query_pitching_by_season(self):
        for range_idx, range_year in enumerate(self.date_ranges):
            range_data = pb.pitching_stats(range_year, qual=-1)
            range_data['Dollars'].apply(remove_dollar_sign)

            for item_data in range_data.values.tolist():
                item_data = [None if check_nan_value(id_) else id_ for id_ in item_data]
                del item_data[51]
                item_data = item_data[1:300]
                self.out_q.put(item_data)
                self.n_items_processed += 1

                if self.n_items_processed % self.summary_every == 0:
                    print('PyBaseballWorker {} pushed {} pitching_by_season items...'.format(self.t_id, self.n_items_processed))
                    print('\tlast year processed {}'.format(range_year))

        self.out_q.put(self.term_item)
        print('* PyBaseballWorker {} pushed term signal for pitching_by_season\n\ttotal items pushed: {}'.format(self.t_id,
                                                                                                       self.n_items_processed))

    def query_batting_by_season(self):
        for range_idx, range_year in enumerate(self.date_ranges):
            range_data = pb.batting_stats(range_year, qual=-1)
            range_data['Dol'].apply(remove_dollar_sign)

            for item_data in range_data.values.tolist():
                item_data = [None if check_nan_value(id_) else id_ for id_ in item_data]
                item_data = item_data[1:288]
                self.out_q.put(item_data)
                self.n_items_processed += 1

                if self.n_items_processed % self.summary_every == 0:
                    print('PyBaseballWorker {} pushed {} batting_by_season items...'.format(self.t_id, self.n_items_processed))
                    print('\tlast year processed {}'.format(range_year))

        # print('* Worker {} pushed term signal for batting_by_season *'.format(self.t_id))
        self.out_q.put(self.term_item)
        print('* PyBaseballWorker {} pushed term signal for batting_by_season\n\ttotal items pushed: {}'.format(self.t_id,
                                                                                                      self.n_items_processed))
