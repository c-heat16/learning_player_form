__author__ = 'Connor Heaton'


import math
import argparse
import os.path

from queue import Queue
from threading import Thread
from datetime import timedelta, date
from sqlite3 import Error, IntegrityError

from SQLWorker import SQLWorker
from PyBaseballWorker import PyBaseballWorker

# supress tqdm in pybaseball calls
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


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


def construct_query_date_range(start_year, start_month, start_day, end_year, range_size):
    start_date = date(start_year, start_month, start_day)
    end_date = date(end_year, 12, 31)

    n_spans = int((end_date - start_date).days) // (range_size)
    date_ranges = []
    for span_idx in range(1, n_spans):
        range_start = start_date + timedelta((span_idx - 1) * range_size)
        range_end = start_date + timedelta((span_idx * range_size) - 1)
        date_ranges.append([range_start, range_end])

    return date_ranges


def check_nan_value(x):
    nan = False
    try:
        nan = math.isnan(x)
    except Exception as ex:
        nan = False

    return nan


def insert_item(conn, query, item):
    cur = conn.cursor()
    success = False
    try:
        cur.execute(query, item)
        conn.commit()
        success = True
    except IntegrityError as e:
        print('Integrity error trying to insert the following item...\n{}'.format(item))
        print('\tError: {}'.format(e))

    return success


def fetch_data(args):

    n_pybaseball_workers = args.n_pybaseball_workers
    print_str = '** n PyBaseball Workers: {} **'.format(n_pybaseball_workers)
    print('*' * len(print_str))
    print(print_str)
    print('*' * len(print_str))

    sql_q = Queue()

    sql_worker = SQLWorker(args=args, in_q=sql_q)

    if sql_worker.connect_to_db():

        if args.statcast:
            query_date_ranges = construct_query_date_range(args.start_year, args.start_month, args.start_day,
                                                           args.end_year, args.n_days_to_query)
            pb_worker_date_ranges = [[] for _ in range(n_pybaseball_workers)]
            for range_idx, qd_range in enumerate(query_date_ranges):
                pb_worker_date_ranges[range_idx % n_pybaseball_workers].append(qd_range)
            print('len(pb_worker_date_ranges): {}'.format(len(pb_worker_date_ranges)))
            print('len(pb_worker_date_ranges[0]): {}'.format(len(pb_worker_date_ranges[0])))

            create_statcast_query_str = """CREATE TABLE IF NOT EXISTS statcast (
                                                            pitch_type TEXT,
                                                            game_date TEXT, 
                                                            release_speed REAL,
                                                            release_pos_x REAL,
                                                            release_pos_z REAL,
                                                            player_name TEXT,
                                                            batter INT,
                                                            pitcher INT,
                                                            events TEXT,
                                                            description TEXT,
                                                            spin_dir BLOB,
                                                            spin_rate_deprecated REAL,
                                                            break_angle_deprecated REAL,
                                                            break_length_deprecated REAL,
                                                            zone INT,
                                                            des TEXT,
                                                            game_type TEXT,
                                                            stand TEXT,
                                                            p_throws TEXT,
                                                            home_team TEXT,
                                                            away_team TEXT,
                                                            type TEXT,
                                                            hit_location INT,
                                                            bb_type TEXT,
                                                            balls INT,
                                                            strikes INT,
                                                            game_year INT,
                                                            pfx_x REAL,
                                                            pfx_z REAL,
                                                            plate_x REAL,
                                                            plate_z REAL,
                                                            on_3b INT,
                                                            on_2b INT,
                                                            on_1b INT,
                                                            outs_when_up INT,
                                                            inning INT,
                                                            inning_topbot TEXT,
                                                            hc_x REAL,
                                                            hc_y REAL,
                                                            tfs_deprecated BLOB,
                                                            tfs_zulu_deprecated BLOB,
                                                            fielder_2 INT,
                                                            umpire BLOB,
                                                            sv_id BLOB,
                                                            vx0 REAL,
                                                            vy0 REAL,
                                                            vz0 REAL,
                                                            ax REAL,
                                                            ay REAL,
                                                            az REAL,
                                                            sz_top REAL,
                                                            sz_bot REAL,
                                                            hit_distance_sc INT,
                                                            launch_speed REAL,
                                                            launch_angle REAL,
                                                            effective_speed REAL,
                                                            release_spin_rate REAL,
                                                            release_extension REAL,
                                                            game_pk INT,
                                                            pitcher_1 INT,
                                                            fielder_2_1 INT,
                                                            fielder_3 INT,
                                                            fielder_4 INT,
                                                            fielder_5 INT,
                                                            fielder_6 INT,
                                                            fielder_7 INT,
                                                            fielder_8 INT,
                                                            fielder_9 INT,
                                                            release_pos_y REAL,
                                                            estimated_ba_using_speedangle REAL,
                                                            estimated_woba_using_speedangle REAL,
                                                            woba_value REAL,
                                                            woba_denom REAL,
                                                            babip_value REAL,
                                                            iso_value REAL,
                                                            launch_speed_angle REAL,
                                                            at_bat_number INT,
                                                            pitch_number INT,
                                                            pitch_name TEXT,
                                                            home_score INT,
                                                            away_score INT,
                                                            bat_score INT,
                                                            fld_score INT,
                                                            post_away_score INT,
                                                            post_home_score INT,
                                                            post_bat_score INT,
                                                            post_fld_score INT,
                                                            if_fielding_alignment TEXT,
                                                            of_fielding_alignment TEXT,
                                                            days_since_2000 INT,
                                                            PRIMARY KEY (game_pk, at_bat_number, pitch_number)
                                                           );"""
            sql_worker.create_table(create_statcast_query_str)

            insert_statcast_query_str = """INSERT INTO statcast (pitch_type,game_date,release_speed,release_pos_x,release_pos_z,player_name,batter,pitcher,events,description,
                                                                                 spin_dir,spin_rate_deprecated,break_angle_deprecated,break_length_deprecated,zone,des,game_type,stand,p_throws,
                                                                                 home_team,away_team,type,hit_location,bb_type,balls,strikes,game_year,pfx_x,pfx_z,plate_x,plate_z,on_3b,on_2b,on_1b,
                                                                                 outs_when_up,inning,inning_topbot,hc_x,hc_y,tfs_deprecated,tfs_zulu_deprecated,fielder_2,umpire,sv_id,vx0,vy0,vz0,
                                                                                 ax,ay,az,sz_top,sz_bot,hit_distance_sc,launch_speed,launch_angle,effective_speed,release_spin_rate,release_extension,
                                                                                 game_pk,pitcher_1,fielder_2_1,fielder_3,fielder_4,fielder_5,fielder_6,fielder_7,fielder_8,fielder_9,release_pos_y,estimated_ba_using_speedangle,
                                                                                 estimated_woba_using_speedangle,woba_value,woba_denom,babip_value,iso_value,launch_speed_angle,at_bat_number,pitch_number,
                                                                                 pitch_name,home_score,away_score,bat_score,fld_score,post_away_score,post_home_score,post_bat_score,post_fld_score,if_fielding_alignment,
                                                                                 of_fielding_alignment, days_since_2000)
                                                                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,julianday(?) - julianday(?))"""
            sql_worker.insert_query = insert_statcast_query_str

            # pb_qs = [Queue() for _ in range(n_pybaseball_workers)]
            pb_workers = [PyBaseballWorker(args=args,
                                           out_q=sql_q,
                                           date_ranges=dr,
                                           t_id=idx) for idx, dr in enumerate(pb_worker_date_ranges)]
            pb_threads = [Thread(target=pbw.query_statcast, args=()) for pbw in pb_workers]
            # sql_thread = Thread(target=sql_worker.insert_items_from_q, args=())

            print('Starting PyBaseball statcast threads...')
            for pb_thread in pb_threads:
                pb_thread.start()

            print('Starting SQL statcast worker...')
            n_inserts = sql_worker.insert_items_from_q(statcast=True)

            print_str = '** Total statcast items inserted: {} **'.format(n_inserts)
            print('*' * len(print_str))
            print(print_str)
            print('*' * len(print_str))

            print('Creating indices on statcast table...')

            print('\tCreating index on game_pk...')
            sql_worker.create_index('CREATE INDEX statcast_game_pk_idx on statcast(game_pk)')
            print('\tCreating index on batters...')
            sql_worker.create_index('CREATE INDEX statcast_batter_idx on statcast(batter)')
            print('\tCreating index on pitchers...')
            sql_worker.create_index('CREATE INDEX statcast_pitcher_idx on statcast(pitcher)')
            print('\tCreating index on game_year...')
            sql_worker.create_index('CREATE INDEX statcast_game_year_idx on statcast(game_year)')
            print('\tCreating index on days_since_2000...')
            sql_worker.create_index('CREATE INDEX statcast_days_since_2000 on statcast(days_since_2000)')

            print('Indices created on statcast table!')

        if args.pitching_by_season:
            create_pitching_by_season_query_str = """CREATE TABLE IF NOT EXISTS pitching_by_season (
                                                                        Season INT,
                                                                        Name TEXT,
                                                                        Team TEXT,
                                                                        Age INT,
                                                                        W INT,
                                                                        L INT,
                                                                        ERA REAL,
                                                                        WAR REAL,
                                                                        G INT,
                                                                        GS INT,
                                                                        CG INT,
                                                                        ShO INT,
                                                                        SV INT, 
                                                                        BS INT,
                                                                        IP REAL,
                                                                        TBF INT,
                                                                        H INT,
                                                                        R INT,
                                                                        ER INT,
                                                                        HR INT,
                                                                        BB INT,
                                                                        IBB INT,
                                                                        HBP INT,
                                                                        WP INT,
                                                                        BK INT,
                                                                        SO INT,
                                                                        GB INT,
                                                                        FB INT,
                                                                        LD INT,
                                                                        IFFB INT,
                                                                        Balls INT,
                                                                        Strikes INT,
                                                                        Pitches INT,
                                                                        RS INT,
                                                                        IFH INT,
                                                                        BU INT,
                                                                        BUH INT,
                                                                        K_9 REAL,
                                                                        BB_9 REAL,
                                                                        K_BB REAL,
                                                                        H_9 REAL,
                                                                        HR_9 REAL,
                                                                        AVG REAL,
                                                                        WHIP REAL,
                                                                        BABIP REAL,
                                                                        LOB_pct REAL,
                                                                        FIP REAL,
                                                                        GB_FB REAL,
                                                                        LD_pct REAL,
                                                                        GB_pct REAL,
                                                                        IFFB_pct REAL,
                                                                        HR_FB REAL,
                                                                        IFH_pct REAL,
                                                                        BUH_pct REAL,
                                                                        Starting REAL,
                                                                        Start_IP REAL,
                                                                        Relieving REAL,
                                                                        Relief_IP REAL,
                                                                        RAR REAL,
                                                                        Dollars REAL,
                                                                        tERA REAL,
                                                                        xFIP REAL,
                                                                        WPA REAL,
                                                                        _WPA REAL,
                                                                        pos_WPA REAL,
                                                                        RE24 REAL,
                                                                        REW REAL,
                                                                        pLI REAL,
                                                                        inLI REAL,
                                                                        gmLI REAL,
                                                                        exLI REAL,
                                                                        Pulls INT,
                                                                        WPA_LI REAL,
                                                                        Clutch REAL,
                                                                        FB_pct REAL,
                                                                        FBv REAL,
                                                                        SL_pct REAL,
                                                                        SLv REAL,
                                                                        CT_pct REAL,
                                                                        CTv REAL,
                                                                        CB_pct REAL,
                                                                        CBv REAL,
                                                                        CH_pct REAL,
                                                                        CHv REAL,
                                                                        SF_pct REAL,
                                                                        SFv REAL,
                                                                        KN_pct REAL,
                                                                        KNv REAL,
                                                                        XX_pct REAL,
                                                                        PO_pct REAL,
                                                                        wFB REAL,
                                                                        wSL REAL,
                                                                        wCT REAL,
                                                                        wCB REAL,
                                                                        wCH REAL,
                                                                        wSF REAL,
                                                                        wKN REAL,
                                                                        wFB_C REAL,
                                                                        wSL_C REAL,
                                                                        wCT_C REAL,
                                                                        wCB_C REAL,
                                                                        wCH_C REAL,
                                                                        wSF_C REAL,
                                                                        wKN_C REAL,
                                                                        O_Swing_pct REAL,
                                                                        Z_Swing_pct REAL,
                                                                        Swing_pct REAL,
                                                                        O_Contact_pct REAL,
                                                                        Z_Contact_pct REAL,
                                                                        Contact_pct REAL,
                                                                        Zone_pct REAL,
                                                                        F_Strike_pct REAL,
                                                                        SwStr_pct REAL,
                                                                        HLD INT,
                                                                        SD INT,
                                                                        MD INT,
                                                                        ERA_n INT,
                                                                        FIP_n INT,
                                                                        xFIP_n INT,
                                                                        K_pct REAL,
                                                                        BB_pct REAL,
                                                                        SIERA REAL,
                                                                        RS_9 REAL,
                                                                        E_F REAL,
                                                                        FA_pct_pfx REAL,
                                                                        FT_pct_pfx REAL,
                                                                        FC_pct_pfx REAL,
                                                                        FS_pct_pfx REAL,
                                                                        FO_pct_pfx REAL,
                                                                        SI_pct_pfx REAL,
                                                                        SL_pct_pfx REAL,
                                                                        CU_pct_pfx REAL,
                                                                        KC_pct_pfx REAL,
                                                                        EP_pct_pfx REAL,
                                                                        CH_pct_pfx REAL,
                                                                        SC_pct_pfx REAL,
                                                                        KN_pct_pfx REAL,
                                                                        UN_pct_pfx REAL,
                                                                        vFA_pfx REAL,
                                                                        vFT_pfx REAL,
                                                                        vFC_pfx REAL,
                                                                        vFS_pfx REAL,
                                                                        vFO_pfx REAL,
                                                                        vSI_pfx REAL,
                                                                        vSL_pfx REAL,
                                                                        vCU_pfx REAL,
                                                                        vKC_pfx REAL,
                                                                        vEP_pfx REAL,
                                                                        vCH_pfx REAL,
                                                                        vSC_pfx REAL,
                                                                        vKN_pfx REAL,
                                                                        FA_X_pfx REAL,
                                                                        FT_X_pfx REAL,
                                                                        FC_X_pfx REAL,
                                                                        FS_X_pfx REAL,
                                                                        FO_X_pfx REAL,
                                                                        SI_X_pfx REAL,
                                                                        SL_X_pfx REAL,
                                                                        CU_X_pfx REAL,
                                                                        KC_X_pfx REAL,
                                                                        EP_X_pfx REAL,
                                                                        CH_X_pfx REAL,
                                                                        SC_X_pfx REAL,
                                                                        KN_X_pfx REAL,
                                                                        FA_Z_pfx REAL,
                                                                        FT_Z_pfx REAL,
                                                                        FC_Z_pfx REAL,
                                                                        FS_Z_pfx REAL,
                                                                        FO_Z_pfx REAL,
                                                                        SI_Z_pfx REAL,
                                                                        SL_Z_pfx REAL,
                                                                        CU_Z_pfx REAL,
                                                                        KC_Z_pfx REAL,
                                                                        EP_Z_pfx REAL,
                                                                        CH_Z_pfx REAL,
                                                                        SC_Z_pfx REAL,
                                                                        KN_Z_pfx REAL,
                                                                        wFA_pfx REAL,
                                                                        wFT_pfx REAL,
                                                                        wFC_pfx REAL,
                                                                        wFS_pfx REAL,
                                                                        wFO_pfx REAL,
                                                                        wSI_pfx REAL,
                                                                        wSL_pfx REAL,
                                                                        wCU_pfx REAL,
                                                                        wKC_pfx REAL,
                                                                        wEP_pfx REAL,
                                                                        wCH_pfx REAL,
                                                                        wSC_pfx REAL,
                                                                        wKN_pfx REAL,
                                                                        wFA_C_pfx REAL,
                                                                        wFT_C_pfx REAL,
                                                                        wFC_C_pfx REAL,
                                                                        wFS_C_pfx REAL,
                                                                        wFO_C_pfx REAL,
                                                                        wSI_C_pfx REAL,
                                                                        wSL_C_pfx REAL,
                                                                        wCU_C_pfx REAL,
                                                                        wKC_C_pfx REAL,
                                                                        wEP_C_pfx REAL,
                                                                        wCH_C_pfx REAL,
                                                                        wSC_C_pfx REAL,
                                                                        wKN_C_pfx REAL,
                                                                        O_Swing_pct_pfx REAL,
                                                                        Z_Swing_pct_pfx REAL,
                                                                        Swing_pct_pfx REAL,
                                                                        O_Contact_pct_pfx REAL,
                                                                        Z_Contact_pct_pfx REAL,
                                                                        Contact_pct_pfx REAL,
                                                                        Zone_pct_pfx REAL,
                                                                        Pace REAL,
                                                                        RA9_WAR REAL,
                                                                        BIP_Wins REAL,
                                                                        LOB_Wins REAL,
                                                                        FDP_Wins REAL,
                                                                        Age_Rng TEXT,
                                                                        K_BB_pct REAL,
                                                                        Pull_pct REAL,
                                                                        Cent_pct REAL,
                                                                        Oppo_pct REAL,
                                                                        Soft_pct REAL,
                                                                        Med_pct REAL,
                                                                        Hard_pct REAL,
                                                                        kwERA REAL,
                                                                        TTO_pct REAL,
                                                                        CH_pct_pi REAL,
                                                                        CS_pct_pi REAL,
                                                                        CU_pct_pi REAL,
                                                                        FA_pct_pi REAL,
                                                                        FC_pct_pi REAL,
                                                                        FS_pct_pi REAL,
                                                                        KN_pct_pi REAL,
                                                                        SB_pct_pi REAL,
                                                                        SI_pct_pi REAL,
                                                                        SL_pct_pi REAL,
                                                                        XX_pct_pi REAL,
                                                                        vCH_pi REAL,
                                                                        vCS_pi REAL,
                                                                        vCU_pi REAL,
                                                                        vFA_pi REAL,
                                                                        vFC_pi REAL,
                                                                        vFS_pi REAL,
                                                                        vKN_pi REAL,
                                                                        vSB_pi REAL,
                                                                        vSI_pi REAL,
                                                                        vSL_pi REAL,
                                                                        vXX_pi REAL,
                                                                        CH_X_pi REAL,
                                                                        CS_X_pi REAL,
                                                                        CU_X_pi REAL,
                                                                        FA_X_pi REAL,
                                                                        FC_X_pi REAL,
                                                                        FS_X_pi REAL,
                                                                        KN_X_pi REAL,
                                                                        SB_X_pi REAL,
                                                                        SI_X_pi REAL,
                                                                        SL_X_pi REAL,
                                                                        XX_X_pi REAL,
                                                                        CH_Z_pi REAL,
                                                                        CS_Z_pi REAL,
                                                                        CU_Z_pi REAL,
                                                                        FA_Z_pi REAL,
                                                                        FC_Z_pi REAL,
                                                                        FS_Z_pi REAL,
                                                                        KN_Z_pi REAL,
                                                                        SB_Z_pi REAL,
                                                                        SI_Z_pi REAL,
                                                                        SL_Z_pi REAL,
                                                                        XX_Z_pi REAL,
                                                                        wCH_pi REAL,
                                                                        wCS_pi REAL,
                                                                        wCU_pi REAL,
                                                                        wFA_pi REAL,
                                                                        wFC_pi REAL,
                                                                        wFS_pi REAL,
                                                                        wKN_pi REAL,
                                                                        wSB_pi REAL,
                                                                        wSI_pi REAL,
                                                                        wSL_pi REAL,
                                                                        wXX_pi REAL,
                                                                        wCH_C_pi REAL,
                                                                        wCS_C_pi REAL,
                                                                        wCU_C_pi REAL,
                                                                        wFA_C_pi REAL,
                                                                        wFC_C_pi REAL,
                                                                        wFS_C_pi REAL,
                                                                        wKN_C_pi REAL,
                                                                        wSB_C_pi REAL,
                                                                        wSI_C_pi REAL,
                                                                        wSL_C_pi REAL,
                                                                        wXX_C_pi REAL,
                                                                        O_Swing_pct_pi REAL,
                                                                        Z_Swing_pct_pi REAL,
                                                                        Swing_pct_pi REAL,
                                                                        O_Contact_pct_pi REAL,
                                                                        Z_Contact_pct_pi REAL,
                                                                        Contact_pct_pi REAL,
                                                                        Zone_pct_pi REAL,
                                                                        Pace_pi REAL,
                                                                        PRIMARY KEY (Season, Name, Team, Age)
                                                                       );"""
            sql_worker.create_table(create_pitching_by_season_query_str)

            insert_pitching_by_season_query_str = """INSERT INTO pitching_by_season (Season,Name,Team,Age,W,L,ERA,WAR,G,GS,CG,ShO,SV,BS,IP,TBF,H,R,ER,HR,BB,IBB,HBP,WP,BK,SO,GB,FB,LD,IFFB,
                                                                                     Balls,Strikes,Pitches,RS,IFH,BU,BUH,K_9,BB_9,K_BB,H_9,HR_9,AVG,WHIP,BABIP,LOB_pct,FIP,GB_FB,LD_pct,
                                                                                     GB_pct,IFFB_pct,HR_FB,IFH_pct,BUH_pct,Starting,Start_IP,Relieving,Relief_IP,RAR,Dollars,tERA,xFIP,
                                                                                     WPA,_WPA,pos_WPA,RE24,REW,pLI,inLI,gmLI,exLI,Pulls,WPA_LI,Clutch,FB_pct,FBv,SL_pct,SLv,CT_pct,CTv,CB_pct,
                                                                                     CBv,CH_pct,CHv,SF_pct,SFv,KN_pct,KNv,XX_pct,PO_pct,wFB,wSL,wCT,wCB,wCH,wSF,wKN,wFB_C,wSL_C,wCT_C,wCB_C,
                                                                                     wCH_C,wSF_C,wKN_C,O_Swing_pct,Z_Swing_pct,Swing_pct,O_Contact_pct,Z_Contact_pct,Contact_pct,Zone_pct,
                                                                                     F_Strike_pct,SwStr_pct,HLD,SD,MD,ERA_n,FIP_n,xFIP_n,K_pct,BB_pct,SIERA,RS_9,E_F,FA_pct_pfx,FT_pct_pfx,FC_pct_pfx,
                                                                                     FS_pct_pfx,FO_pct_pfx,SI_pct_pfx,SL_pct_pfx,CU_pct_pfx,KC_pct_pfx,EP_pct_pfx,CH_pct_pfx,SC_pct_pfx,KN_pct_pfx,
                                                                                     UN_pct_pfx,vFA_pfx,vFT_pfx,vFC_pfx,vFS_pfx,vFO_pfx,vSI_pfx,vSL_pfx,vCU_pfx,vKC_pfx,vEP_pfx,vCH_pfx,vSC_pfx,
                                                                                     vKN_pfx,FA_X_pfx,FT_X_pfx,FC_X_pfx,FS_X_pfx,FO_X_pfx,SI_X_pfx,SL_X_pfx,CU_X_pfx,KC_X_pfx,EP_X_pfx,CH_X_pfx,
                                                                                     SC_X_pfx,KN_X_pfx,FA_Z_pfx,FT_Z_pfx,FC_Z_pfx,FS_Z_pfx,FO_Z_pfx,SI_Z_pfx,SL_Z_pfx,CU_Z_pfx,KC_Z_pfx,EP_Z_pfx,
                                                                                     CH_Z_pfx,SC_Z_pfx,KN_Z_pfx,wFA_pfx,wFT_pfx,wFC_pfx,wFS_pfx,wFO_pfx,wSI_pfx,wSL_pfx,wCU_pfx,wKC_pfx,wEP_pfx,wCH_pfx,
                                                                                     wSC_pfx,wKN_pfx,wFA_C_pfx,wFT_C_pfx,wFC_C_pfx,wFS_C_pfx,wFO_C_pfx,wSI_C_pfx,wSL_C_pfx,wCU_C_pfx,wKC_C_pfx,wEP_C_pfx,
                                                                                     wCH_C_pfx,wSC_C_pfx,wKN_C_pfx,O_Swing_pct_pfx,Z_Swing_pct_pfx,Swing_pct_pfx,O_Contact_pct_pfx,Z_Contact_pct_pfx,
                                                                                     Contact_pct_pfx,Zone_pct_pfx,Pace,RA9_WAR,BIP_Wins,LOB_Wins,FDP_Wins,Age_Rng,K_BB_pct,Pull_pct,Cent_pct,Oppo_pct,
                                                                                     Soft_pct,Med_pct,Hard_pct,kwERA,TTO_pct,CH_pct_pi,CS_pct_pi,CU_pct_pi,FA_pct_pi,FC_pct_pi,FS_pct_pi,KN_pct_pi,SB_pct_pi,
                                                                                     SI_pct_pi,SL_pct_pi,XX_pct_pi,vCH_pi,vCS_pi,vCU_pi,vFA_pi,vFC_pi,vFS_pi,vKN_pi,vSB_pi,vSI_pi,vSL_pi,vXX_pi,CH_X_pi,
                                                                                     CS_X_pi,CU_X_pi,FA_X_pi,FC_X_pi,FS_X_pi,KN_X_pi,SB_X_pi,SI_X_pi,SL_X_pi,XX_X_pi,CH_Z_pi,CS_Z_pi,CU_Z_pi,FA_Z_pi,
                                                                                     FC_Z_pi,FS_Z_pi,KN_Z_pi,SB_Z_pi,SI_Z_pi,SL_Z_pi,XX_Z_pi,wCH_pi,wCS_pi,wCU_pi,wFA_pi,wFC_pi,wFS_pi,wKN_pi,wSB_pi,
                                                                                     wSI_pi,wSL_pi,wXX_pi,wCH_C_pi,wCS_C_pi,wCU_C_pi,wFA_C_pi,wFC_C_pi,wFS_C_pi,wKN_C_pi,wSB_C_pi,wSI_C_pi,wSL_C_pi,
                                                                                     wXX_C_pi,O_Swing_pct_pi,Z_Swing_pct_pi,Swing_pct_pi,O_Contact_pct_pi,Z_Contact_pct_pi,Contact_pct_pi,Zone_pct_pi,Pace_pi)
                                                                                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
            sql_worker.insert_query = insert_pitching_by_season_query_str

            pitching_by_season_year_range = [i for i in range(args.start_year, args.end_year + 1)]
            pbs_worker_date_ranges = [[] for _ in range(n_pybaseball_workers)]
            for season_idx, season_year in enumerate(pitching_by_season_year_range):
                pbs_worker_date_ranges[season_idx % n_pybaseball_workers].append(season_year)

            pb_workers = [PyBaseballWorker(args=args,
                                           out_q=sql_q,
                                           date_ranges=dr,
                                           t_id=idx) for idx, dr in enumerate(pbs_worker_date_ranges)]
            pb_threads = [Thread(target=pbw.query_pitching_by_season, args=()) for pbw in pb_workers]

            print('Starting PyBaseball pitching_by_season threads...')
            for pb_thread in pb_threads:
                pb_thread.start()

            print('Starting SQL pitching_by_season worker...')
            # sql_thread.start()
            n_inserts = sql_worker.insert_items_from_q()

            print_str = '** Total pitching_by_season items inserted: {} **'.format(n_inserts)
            print('*' * len(print_str))
            print(print_str)
            print('*' * len(print_str))

            print('Creating indices on pitching_by_season table...')

            print('\tCreating index on player names...')
            sql_worker.create_index('CREATE INDEX pitching_name_idx on pitching_by_season(Name)')
            print('\tCreating index on season...')
            sql_worker.create_index('CREATE INDEX pitching_season_idx on pitching_by_season(Season)')

            print('Indices created on pitching_by_season_table!')

        if args.batting_by_season:
            create_batting_by_season_query_str = """CREATE TABLE IF NOT EXISTS batting_by_season (
                                                                                    Season INT,
                                                                                    Name TEXT,
                                                                                    Team TEXT,
                                                                                    Age INT,
                                                                                    G INT,
                                                                                    AB INT,
                                                                                    PA INT,
                                                                                    H INT,
                                                                                    B1 INT,
                                                                                    B2 INT,
                                                                                    B3 INT,
                                                                                    HR INT,
                                                                                    R INT,
                                                                                    RBI INT,
                                                                                    BB INT,
                                                                                    IBB INT,
                                                                                    SO INT,
                                                                                    HBP INT,
                                                                                    SF INT,
                                                                                    SH INT,
                                                                                    GDP INT,
                                                                                    SB INT,
                                                                                    CS INT,
                                                                                    AVG REAL,
                                                                                    GB INT,
                                                                                    FB INT,
                                                                                    LD INT,
                                                                                    IFFB INT,
                                                                                    Pitches INT,
                                                                                    Balls INT,
                                                                                    Strikes INT,
                                                                                    IFH INT,
                                                                                    BU INT,
                                                                                    BUH INT,
                                                                                    BB_pct REAL,
                                                                                    K_pct REAL,
                                                                                    BB_K REAL,
                                                                                    OBP REAL,
                                                                                    SLG REAL,
                                                                                    OPS REAL,
                                                                                    ISO REAL,
                                                                                    BABIP REAL,
                                                                                    GB_FB REAL,
                                                                                    LD_pct REAL,
                                                                                    GB_pct REAL,
                                                                                    FB_pct REAL,
                                                                                    IFFB_pct REAL,
                                                                                    HR_FB REAL,
                                                                                    IFH_pct REAL,
                                                                                    BUH_pct REAL,
                                                                                    wOBA REAL,
                                                                                    wRAA REAL,
                                                                                    wRC REAL,
                                                                                    Bat REAL,
                                                                                    Fld REAL,
                                                                                    Rep REAL,
                                                                                    Pos REAL,
                                                                                    RAR REAL,
                                                                                    WAR REAL,
                                                                                    Dol REAL,
                                                                                    Spd REAL,
                                                                                    wRC_pos INT,
                                                                                    WPA REAL,
                                                                                    _WPA REAL,
                                                                                    pos_WPA REAL,
                                                                                    RE24 REAL,
                                                                                    REW REAL,
                                                                                    pLI REAL,
                                                                                    phLI REAL,
                                                                                    PH INT,
                                                                                    WPA_LI REAL,
                                                                                    Clutch REAL,
                                                                                    FB_pct_Pitch REAL,
                                                                                    FBv REAL,
                                                                                    SL_pct REAL,
                                                                                    SLv REAL,
                                                                                    CT_pct REAL,
                                                                                    CTv REAL,
                                                                                    CB_pct REAL,
                                                                                    CBv REAL,
                                                                                    CH_pct REAL,
                                                                                    CHv REAL,
                                                                                    SF_pct REAL,
                                                                                    SFv REAL,
                                                                                    KN_pct REAL,
                                                                                    KNv REAL,
                                                                                    XX_pct REAL,
                                                                                    PO_pct REAL,
                                                                                    wFB REAL,
                                                                                    wSL REAL,
                                                                                    wCT REAL,
                                                                                    wCB REAL,
                                                                                    wCH REAL,
                                                                                    wSF REAL,
                                                                                    wKN REAL,
                                                                                    wFB_C REAL,
                                                                                    wSL_C REAL,
                                                                                    wCT_C REAL,
                                                                                    wCB_C REAL,
                                                                                    wCH_C REAL,
                                                                                    wSF_C REAL,
                                                                                    wKN_C REAL,
                                                                                    O_Swing_pct REAL,
                                                                                    Z_Swing_pct REAL,
                                                                                    Swing_pct REAL,
                                                                                    O_Contact_pct REAL,
                                                                                    Z_Contact_pct REAL,
                                                                                    Contact_pct REAL,
                                                                                    Zone_pct REAL,
                                                                                    F_Strike_pct REAL,
                                                                                    SwStr_pct REAL,
                                                                                    BsR REAL,
                                                                                    FA_pct_pfx REAL,
                                                                                    FT_pct_pfx REAL,
                                                                                    FC_pct_pfx REAL,
                                                                                    FS_pct_pfx REAL,
                                                                                    FO_pct_pfx REAL,
                                                                                    SI_pct_pfx REAL,
                                                                                    SL_pct_pfx REAL,
                                                                                    CU_pct_pfx REAL,
                                                                                    KC_pct_pfx REAL,
                                                                                    EP_pct_pfx REAL,
                                                                                    CH_pct_pfx REAL,
                                                                                    SC_pct_pfx REAL,
                                                                                    KN_pct_pfx REAL,
                                                                                    UN_pct_pfx REAL,
                                                                                    vFA_pfx REAL,
                                                                                    vFT_pfx REAL,
                                                                                    vFC_pfx REAL,
                                                                                    vFS_pfx REAL,
                                                                                    vFO_pfx REAL,
                                                                                    vSI_pfx REAL,
                                                                                    vSL_pfx REAL,
                                                                                    vCU_pfx REAL,
                                                                                    vKC_pfx REAL,
                                                                                    vEP_pfx REAL,
                                                                                    vCH_pfx REAL,
                                                                                    vSC_pfx REAL,
                                                                                    vKN_pfx REAL,
                                                                                    FA_X_pfx REAL,
                                                                                    FT_X_pfx REAL,
                                                                                    FC_X_pfx REAL,
                                                                                    FS_X_pfx REAL,
                                                                                    FO_X_pfx REAL,
                                                                                    SI_X_pfx REAL,
                                                                                    SL_X_pfx REAL,
                                                                                    CU_X_pfx REAL,
                                                                                    KC_X_pfx REAL,
                                                                                    EP_X_pfx REAL,
                                                                                    CH_X_pfx REAL,
                                                                                    SC_X_pfx REAL,
                                                                                    KN_X_pfx REAL,
                                                                                    FA_Z_pfx REAL,
                                                                                    FT_Z_pfx REAL,
                                                                                    FC_Z_pfx REAL,
                                                                                    FS_Z_pfx REAL,
                                                                                    FO_Z_pfx REAL,
                                                                                    SI_Z_pfx REAL,
                                                                                    SL_Z_pfx REAL,
                                                                                    CU_Z_pfx REAL,
                                                                                    KC_Z_pfx REAL,
                                                                                    EP_Z_pfx REAL,
                                                                                    CH_Z_pfx REAL,
                                                                                    SC_Z_pfx REAL,
                                                                                    KN_Z_pfx REAL,
                                                                                    wFA_pfx REAL,
                                                                                    wFT_pfx REAL,
                                                                                    wFC_pfx REAL,
                                                                                    wFS_pfx REAL,
                                                                                    wFO_pfx REAL,
                                                                                    wSI_pfx REAL,
                                                                                    wSL_pfx REAL,
                                                                                    wCU_pfx REAL,
                                                                                    wKC_pfx REAL,
                                                                                    wEP_pfx REAL,
                                                                                    wCH_pfx REAL,
                                                                                    wSC_pfx REAL,
                                                                                    wKN_pfx REAL,
                                                                                    wFA_C_pfx REAL,
                                                                                    wFT_C_pfx REAL,
                                                                                    wFC_C_pfx REAL,
                                                                                    wFS_C_pfx REAL,
                                                                                    wFO_C_pfx REAL,
                                                                                    wSI_C_pfx REAL,
                                                                                    wSL_C_pfx REAL,
                                                                                    wCU_C_pfx REAL,
                                                                                    wKC_C_pfx REAL,
                                                                                    wEP_C_pfx REAL,
                                                                                    wCH_C_pfx REAL,
                                                                                    wSC_C_pfx REAL,
                                                                                    wKN_C_pfx REAL,
                                                                                    O_Swing_pct_pfx REAL,
                                                                                    Z_Swing_pct_pfx REAL,
                                                                                    Swing_pct_pfx REAL,
                                                                                    O_Contact_pct_pfx REAL,
                                                                                    Z_Contact_pct_pfx REAL,
                                                                                    Contact_pct_pfx REAL,
                                                                                    Zone_pct_pfx REAL,
                                                                                    Pace REAL,
                                                                                    Def REAL,
                                                                                    wSB REAL,
                                                                                    UBR REAL,
                                                                                    Age_Rng TEXT,
                                                                                    Off REAL,
                                                                                    Lg REAL,
                                                                                    wGDP REAL,
                                                                                    Pull_pct REAL,
                                                                                    Cent_pct REAL,
                                                                                    Oppo_pct REAL,
                                                                                    Soft_pct REAL,
                                                                                    Med_pct REAL,
                                                                                    Hard_pct REAL,
                                                                                    TTO_pct REAL,
                                                                                    CH_pct_pi REAL,
                                                                                    CS_pct_pi REAL,
                                                                                    CU_pct_pi REAL,
                                                                                    FA_pct_pi REAL,
                                                                                    FC_pct_pi REAL,
                                                                                    FS_pct_pi REAL,
                                                                                    KN_pct_pi REAL,
                                                                                    SB_pct_pi REAL,
                                                                                    SI_pct_pi REAL,
                                                                                    SL_pct_pi REAL,
                                                                                    XX_pct_pi REAL,
                                                                                    vCH_pi REAL,
                                                                                    vCS_pi REAL,
                                                                                    vCU_pi REAL,
                                                                                    vFA_pi REAL,
                                                                                    vFC_pi REAL,
                                                                                    vFS_pi REAL,
                                                                                    vKN_pi REAL,
                                                                                    vSB_pi REAL,
                                                                                    vSI_pi REAL,
                                                                                    vSL_pi REAL,
                                                                                    vXX_pi REAL,
                                                                                    CH_X_pi REAL,
                                                                                    CS_X_pi REAL,
                                                                                    CU_X_pi REAL,
                                                                                    FA_X_pi REAL,
                                                                                    FC_X_pi REAL,
                                                                                    FS_X_pi REAL,
                                                                                    KN_X_pi REAL,
                                                                                    SB_X_pi REAL,
                                                                                    SI_X_pi REAL,
                                                                                    SL_X_pi REAL,
                                                                                    XX_X_pi REAL,
                                                                                    CH_Z_pi REAL,
                                                                                    CS_Z_pi REAL,
                                                                                    CU_Z_pi REAL,
                                                                                    FA_Z_pi REAL,
                                                                                    FC_Z_pi REAL,
                                                                                    FS_Z_pi REAL,
                                                                                    KN_Z_pi REAL,
                                                                                    SB_Z_pi REAL,
                                                                                    SI_Z_pi REAL,
                                                                                    SL_Z_pi REAL,
                                                                                    XX_Z_pi REAL,
                                                                                    wCH_pi REAL,
                                                                                    wCS_pi REAL,
                                                                                    wCU_pi REAL,
                                                                                    wFA_pi REAL,
                                                                                    wFC_pi REAL,
                                                                                    wFS_pi REAL,
                                                                                    wKN_pi REAL,
                                                                                    wSB_pi REAL,
                                                                                    wSI_pi REAL,
                                                                                    wSL_pi REAL,
                                                                                    wXX_pi REAL,
                                                                                    wCH_C_pi REAL,
                                                                                    wCS_C_pi REAL,
                                                                                    wCU_C_pi REAL,
                                                                                    wFA_C_pi REAL,
                                                                                    wFC_C_pi REAL,
                                                                                    wFS_C_pi REAL,
                                                                                    wKN_C_pi REAL,
                                                                                    wSB_C_pi REAL,
                                                                                    wSI_C_pi REAL,
                                                                                    wSL_C_pi REAL,
                                                                                    wXX_C_pi REAL,
                                                                                    O_Swing_pct_pi REAL,
                                                                                    Z_Swing_pct_pi REAL,
                                                                                    Swing_pct_pi REAL,
                                                                                    O_Contact_pct_pi REAL,
                                                                                    Z_Contact_pct_pi REAL,
                                                                                    Contact_pct_pi REAL,
                                                                                    Zone_pct_pi REAL,
                                                                                    Pace_pi REAL,
                                                                                    PRIMARY KEY (Season, Name, Team, Age)
                                                                                   );"""
            sql_worker.create_table(create_batting_by_season_query_str)

            insert_batting_by_season_query_str = """INSERT INTO batting_by_season (Season,Name,Team,Age,G,AB,PA,H,B1,B2,B3,HR,R,RBI,BB,IBB,SO,HBP,SF,SH,GDP,SB,CS,
                                                    AVG,GB,FB,LD,IFFB,Pitches,Balls,Strikes,IFH,BU,BUH,BB_pct,K_pct,BB_K,OBP,SLG,OPS,ISO,BABIP,GB_FB,LD_pct,GB_pct,
                                                    FB_pct,IFFB_pct,HR_FB,IFH_pct,BUH_pct,wOBA,wRAA,wRC,Bat,Fld,Rep,Pos,RAR,WAR,Dol,Spd,wRC_pos,WPA,_WPA,pos_WPA,
                                                    RE24,REW,pLI,phLI,PH,WPA_LI,Clutch,FB_pct_Pitch,FBv,SL_pct,SLv,CT_pct,CTv,CB_pct,CBv,CH_pct,CHv,SF_pct,SFv,
                                                    KN_pct,KNv,XX_pct,PO_pct,wFB,wSL,wCT,wCB,wCH,wSF,wKN,wFB_C,wSL_C,wCT_C,wCB_C,wCH_C,wSF_C,wKN_C,O_Swing_pct,
                                                    Z_Swing_pct,Swing_pct,O_Contact_pct,Z_Contact_pct,Contact_pct,Zone_pct,F_Strike_pct,SwStr_pct,BsR,FA_pct_pfx,
                                                    FT_pct_pfx,FC_pct_pfx,FS_pct_pfx,FO_pct_pfx,SI_pct_pfx,SL_pct_pfx,CU_pct_pfx,KC_pct_pfx,EP_pct_pfx,CH_pct_pfx,
                                                    SC_pct_pfx,KN_pct_pfx,UN_pct_pfx,vFA_pfx,vFT_pfx,vFC_pfx,vFS_pfx,vFO_pfx,vSI_pfx,vSL_pfx,vCU_pfx,vKC_pfx,
                                                    vEP_pfx,vCH_pfx,vSC_pfx,vKN_pfx,FA_X_pfx,FT_X_pfx,FC_X_pfx,FS_X_pfx,FO_X_pfx,SI_X_pfx,SL_X_pfx,CU_X_pfx,
                                                    KC_X_pfx,EP_X_pfx,CH_X_pfx,SC_X_pfx,KN_X_pfx,FA_Z_pfx,FT_Z_pfx,FC_Z_pfx,FS_Z_pfx,FO_Z_pfx,SI_Z_pfx,SL_Z_pfx,
                                                    CU_Z_pfx,KC_Z_pfx,EP_Z_pfx,CH_Z_pfx,SC_Z_pfx,KN_Z_pfx,wFA_pfx,wFT_pfx,wFC_pfx,wFS_pfx,wFO_pfx,wSI_pfx,wSL_pfx,
                                                    wCU_pfx,wKC_pfx,wEP_pfx,wCH_pfx,wSC_pfx,wKN_pfx,wFA_C_pfx,wFT_C_pfx,wFC_C_pfx,wFS_C_pfx,wFO_C_pfx,wSI_C_pfx,
                                                    wSL_C_pfx,wCU_C_pfx,wKC_C_pfx,wEP_C_pfx,wCH_C_pfx,wSC_C_pfx,wKN_C_pfx,O_Swing_pct_pfx,Z_Swing_pct_pfx,
                                                    Swing_pct_pfx,O_Contact_pct_pfx,Z_Contact_pct_pfx,Contact_pct_pfx,Zone_pct_pfx,Pace,Def,wSB,UBR,Age_Rng,
                                                    Off,Lg,wGDP,Pull_pct,Cent_pct,Oppo_pct,Soft_pct,Med_pct,Hard_pct,TTO_pct,CH_pct_pi,CS_pct_pi,CU_pct_pi,
                                                    FA_pct_pi,FC_pct_pi,FS_pct_pi,KN_pct_pi,SB_pct_pi,SI_pct_pi,SL_pct_pi,XX_pct_pi,vCH_pi,vCS_pi,vCU_pi,vFA_pi,
                                                    vFC_pi,vFS_pi,vKN_pi,vSB_pi,vSI_pi,vSL_pi,vXX_pi,CH_X_pi,CS_X_pi,CU_X_pi,FA_X_pi,FC_X_pi,FS_X_pi,KN_X_pi,SB_X_pi,
                                                    SI_X_pi,SL_X_pi,XX_X_pi,CH_Z_pi,CS_Z_pi,CU_Z_pi,FA_Z_pi,FC_Z_pi,FS_Z_pi,KN_Z_pi,SB_Z_pi,SI_Z_pi,SL_Z_pi,XX_Z_pi,
                                                    wCH_pi,wCS_pi,wCU_pi,wFA_pi,wFC_pi,wFS_pi,wKN_pi,wSB_pi,wSI_pi,wSL_pi,wXX_pi,wCH_C_pi,wCS_C_pi,wCU_C_pi,wFA_C_pi,
                                                    wFC_C_pi,wFS_C_pi,wKN_C_pi,wSB_C_pi,wSI_C_pi,wSL_C_pi,wXX_C_pi,O_Swing_pct_pi,Z_Swing_pct_pi,Swing_pct_pi,
                                                    O_Contact_pct_pi,Z_Contact_pct_pi,Contact_pct_pi,Zone_pct_pi,Pace_pi)
                                                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
            sql_worker.insert_query = insert_batting_by_season_query_str
            pitching_by_season_year_range = [i for i in range(args.start_year, args.end_year + 1)]
            pbs_worker_date_ranges = [[] for _ in range(n_pybaseball_workers)]
            for season_idx, season_year in enumerate(pitching_by_season_year_range):
                pbs_worker_date_ranges[season_idx % n_pybaseball_workers].append(season_year)

            pb_workers = [PyBaseballWorker(args=args,
                                           out_q=sql_q,
                                           date_ranges=dr,
                                           t_id=idx) for idx, dr in enumerate(pbs_worker_date_ranges)]
            pb_threads = [Thread(target=pbw.query_batting_by_season, args=()) for pbw in pb_workers]

            print('Starting PyBaseball batting_by_season threads...')
            for pb_thread in pb_threads:
                pb_thread.start()

            print('Starting SQL batting_by_season worker...')
            # sql_thread.start()
            n_inserts = sql_worker.insert_items_from_q()

            print_str = '** Total batting_by_season items inserted: {} **'.format(n_inserts)
            print('*' * len(print_str))
            print(print_str)
            print('*' * len(print_str))

            print('Creating indices on batting_by_season table...')

            print('\tCreating index on player names...')
            sql_worker.create_index('CREATE INDEX batting_name_idx on batting_by_season(Name)')
            print('\tCreating index on season...')
            sql_worker.create_index('CREATE INDEX batting_season_idx on batting_by_season(Season)')

            print('Indices created on batting_by_season table!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--statcast', default=True, type=str2bool)
    parser.add_argument('--pitching_by_season', default=False, type=str2bool)
    parser.add_argument('--batting_by_season', default=False, type=str2bool)

    parser.add_argument('--start_year', default=2015, type=int)
    parser.add_argument('--start_month', default=1, type=int)
    parser.add_argument('--start_day', default=1, type=int)
    parser.add_argument('--end_year', default=2019, type=int)
    parser.add_argument('--n_days_to_query', default=3, type=int)
    parser.add_argument('--pb_summary_every', default=10000, type=int)
    parser.add_argument('--sql_summary_every', default=10000, type=int)
    parser.add_argument('--sql_insert_size', default=250, type=int)
    parser.add_argument('--database_fp', default='../database/mlb.db')
    parser.add_argument('--term_item', default='<END>')
    parser.add_argument('--sql_n_sleep', default=5, type=int)

    parser.add_argument('--n_pybaseball_workers', default=1, type=int)

    args = parser.parse_args()

    db_basedir = os.path.split(args.database_fp)[0]
    if not os.path.exists(db_basedir):
        os.makedirs(db_basedir)

    print('Fetching data...')
    fetch_data(args)
