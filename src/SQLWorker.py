__author__ = 'Connor Heaton'


import time
import sqlite3
from sqlite3 import Error, IntegrityError


class SQLWorker(object):
    def __init__(self, args, in_q):
        self.args = args
        self.in_q = in_q

        self.database_fp = self.args.database_fp
        self.n_sleep = self.args.sql_n_sleep
        self.term_item = self.args.term_item
        self.n_pybaseball_workers = self.args.n_pybaseball_workers
        self.summary_every = self.args.sql_summary_every
        self.sql_insert_size = self.args.sql_insert_size

        self.conn = None
        self.insert_query = None

    def connect_to_db(self):
        success = False
        try:
            self.conn = sqlite3.connect(self.database_fp)
            # self.conn.execute('PRAGMA synchronous = OFF')
            print('sqlite3.version: {}'.format(sqlite3.version))
            success = True
        except Error as e:
            print('Error connecting to SQL DB: {}'.format(e))

        return success

    def create_table(self, query_str):
        success = False

        try:
            c = self.conn.cursor()
            c.execute(query_str)
            success = True
        except Error as e:
            print('Error creating db: {}'.format(e))

        return success

    def insert_items_from_q(self, statcast=False):
        self.conn = sqlite3.connect(self.database_fp)
        # insert_size = 100
        items_to_insert = []

        print_str = '* SQLWorker.insert_size: {} *'.format(self.sql_insert_size)
        print('*' * len(print_str))
        print(print_str)
        print('*' * len(print_str))

        n_inserted = 0
        n_term_rcvd = 0
        print('SQLWorker sleeping before starting...')
        time.sleep(5)
        print('SQLWorker starting now...')

        while True:
            if self.in_q.empty():
                print('SQLWorker in q empty... sleeping for {} seconds...'.format(self.n_sleep))
                time.sleep(self.n_sleep)
            else:
                in_data = self.in_q.get()

                if in_data == self.term_item:
                    n_term_rcvd += 1
                    print('* SQLWorker received term signal (total={}, max={}) *'.format(n_term_rcvd,
                                                                                         self.n_pybaseball_workers))
                    if n_term_rcvd == self.n_pybaseball_workers:
                        print('* SQLWorker received all term signals (n={})... stopping *'.format(n_term_rcvd))

                        if len(items_to_insert) >= 1:
                            insert_success = self.insert_items(query_str=self.insert_query, items=items_to_insert)

                            if insert_success:
                                n_inserted += len(items_to_insert)
                        break
                else:
                    if statcast:
                        in_data.extend([in_data[1], '2000-01-01'])
                    items_to_insert.append(in_data)

                    if len(items_to_insert) >= self.sql_insert_size:
                        insert_success = self.insert_items(query_str=self.insert_query, items=items_to_insert)
                        items_to_insert = []

                        if insert_success:
                            n_inserted += self.sql_insert_size

                            if n_inserted % self.summary_every == 0:
                                print('SQLWorker inserted {} items...'.format(n_inserted))

        return n_inserted

    def insert_item(self, query_str, item):
        success = False
        c = self.conn.cursor()

        try:
            c.execute(query_str, item)
            self.conn.commit()
            success = True
        except IntegrityError as e:
            print('Integrity error trying to insert the following item...\n{}'.format(item))
            print('\tError: {}'.format(e))

        return success

    def insert_items(self, query_str, items):
        success = False
        c = self.conn.cursor()

        try:
            c.executemany(query_str, items)
            self.conn.commit()
            success = True
        except IntegrityError as e:
            print('Integrity error trying to insert the following items...\n{}'.format(items))
            print('\tError: {}'.format(e))

        return success

    def create_index(self, index_str):
        c = self.conn.cursor()
        c.execute(index_str)
        self.conn.commit()


