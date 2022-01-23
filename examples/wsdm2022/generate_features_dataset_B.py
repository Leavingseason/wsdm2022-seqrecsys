from collections import defaultdict
import os
import random
from unicodedata import normalize 
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time
import sys

def get_hour_and_weekday(timestamp):
    d = datetime.fromtimestamp(timestamp)
    hour = int(d.hour)
    day = int(d.weekday())
    return (hour, day)

def load_user_order_history(infile):
    user2history = {} 
    edge_type2freq = defaultdict(int)
    item2history = {} 
    with open(infile, 'r') as rd:
        cnt = 0
        while True:
            line = rd.readline()
            if not line:
                break
            if cnt % 10000 == 0:
                print('\r loading line {0}w'.format(cnt//10000), end=' ')
            cnt += 1
            words = line[:-1].split(',')
            userid = int(words[0])
            itemid = int(words[1])
            edge_type = int(words[2])
            timestamp = int(words[3]) 
            edge_type2freq[edge_type] += 1
            if userid not in user2history:
                user2history[userid] = defaultdict(list)             
            user2history[userid][itemid].append((edge_type, timestamp, *(get_hour_and_weekday(timestamp))))
            if itemid not in item2history:
                item2history[itemid] = defaultdict(list)
            item2history[itemid][userid].append((edge_type, timestamp, *(get_hour_and_weekday(timestamp))))
    
    print('#.users = {0}'.format(len(user2history)))
    return user2history, item2history, edge_type2freq

def collide(element, itemid, edge_type, timestamp, tol=40):
    if element[0] == itemid and element[1] == edge_type and abs((element[2]-timestamp)/(60*60)) <= tol:
        return True
    else:
        return False



def concat_elements(t, length, idx, max_hist_len):

    if length <=0:
        return '-1'
    if length > max_hist_len:
        start = length - max_hist_len
    else:
        start = 0
    res = ','.join([str(e[idx]) for e in t[start:length]])
    return res
 
def generate_features(infile, outfile, user2history, item2history, edge_types=[]):
    print('processing file {0}'.format(os.path.basename(infile)))
    rd = open(infile, 'r') 
    wt = open(outfile, 'w')

    N_edge_type = len(edge_types)
    user_time2cnt = {}
    user2hour_dist = {}
    user2day_dist = {}
    item2follow_stat = {}


    cnt = 0
    t0 = time.time()
    while True:
        line = rd.readline()
        if not line:
            break
        if cnt % 1000 == 0:
            t1 = time.time()
            print('\rprocess line {0} k, time elapse {1:.2f} m'.format(cnt // 1000, (t1-t0)/60), end=' ')
        cnt += 1
        words = line[:-1].split('\t')
        userid, itemid, edgetype, timestamp = int(words[1]), int(words[2]), int(words[3]), int(words[4])
        hour, day = get_hour_and_weekday(timestamp)
        features = []
        user_cnt = 0
        user_item_cnt = 0
        user_item_ratio = 0
        user_item_edgetype_cnt = 0
        user_hour_dist = [0] * 24
        user_day_dist = [0] * 7
        user_item_hour_dist = [0] * 24
        user_item_day_dist = [0] * 7
        user_item_targethour_cnt = 0
        user_item_taragetday_cnt = 0
        user_item_edge_targethour_cnt = 0
        user_item_edge_targetday_cnt = 0

        
        item_follow_cnt = 0
        item_follow_edgetype_cnt = 0
        item_follow_hour_cnt = 0
        item_follow_day_cnt = 0 
        item_follow_edgetype_hour_cnt = 0
        item_follow_edgetype_day_cnt = 0

        if userid in user2history:
            if itemid in user2history[userid]:                
                for t in user2history[userid][itemid]:
                    if t[1] < timestamp:
                        user_item_cnt += 1

                        user_item_hour_dist[hour] += 1
                        user_item_day_dist[day] += 1

                        if hour == t[2]:
                            user_item_targethour_cnt += 1
                        if day == t[3]:
                            user_item_taragetday_cnt += 1

                        if t[0] == edgetype:
                            user_item_edgetype_cnt += 1
                            if hour == t[2]:
                                user_item_edge_targethour_cnt += 1
                            if day == t[3]:
                                user_item_edge_targetday_cnt += 1

            user_item_key = '{0}_{1}'.format(userid, timestamp)

            if user_item_key in user_time2cnt:
                user_cnt = user_time2cnt[user_item_key]                
            else:
                for itemid, hist in user2history[userid].items():
                    for t in hist:
                        if t[1] < timestamp:
                            user_cnt += 1
                user_time2cnt[user_item_key] = user_cnt
            
            if userid in user2hour_dist:
                user_hour_dist = user2hour_dist[userid]
                user_day_dist = user2day_dist[userid]
            else:
                for _, hist in user2history[userid].items():
                    for t in hist:
                        user_hour_dist[t[2]] += 1
                        user_day_dist[t[3]] += 1
                user2hour_dist[userid] = list(map(max, user_hour_dist, [5]*24))
                user2day_dist[userid] = list(map(max, user_day_dist, [5]*24))
            
            if user_cnt > 0:
                user_item_ratio = user_item_cnt * 1.0 / user_cnt
        
        if itemid in item2follow_stat:
            item_follow_cnt, item_follow_hour_cnt, item_follow_day_cnt, item_follow_edgetype_cnt, item_follow_edgetype_hour_cnt, item_follow_edgetype_day_cnt = item2follow_stat[itemid]
        else:
            if itemid in item2history:
                for u, hist in item2history[itemid].items():
                    for t in hist:
                        if True: #t[1] < timestamp:
                            item_follow_cnt += 1

                            if t[2] == hour:
                                item_follow_hour_cnt += 1
                            if t[3] == day:
                                item_follow_day_cnt += 1

                            if t[0] == edgetype:
                                item_follow_edgetype_cnt += 1
                                if t[2] == hour:
                                    item_follow_edgetype_hour_cnt += 1 
                                if t[3] == day:
                                    item_follow_edgetype_day_cnt += 1
            item2follow_stat[itemid] = max(0, item_follow_cnt - 10), max(0, item_follow_hour_cnt - 10), max(0, item_follow_day_cnt - 10), max(0, item_follow_edgetype_cnt - 10), max(0, item_follow_edgetype_hour_cnt - 10), max(0, item_follow_edgetype_day_cnt - 10)
        features = [
            user_cnt,
            user_item_cnt ,
            user_item_ratio ,
            user_item_edgetype_cnt,
            *user_item_hour_dist ,
            *user_item_day_dist,
            user_item_targethour_cnt,
            user_item_taragetday_cnt,
            user_item_edge_targethour_cnt,
            user_item_edge_targetday_cnt ,
            item_follow_cnt ,
            item_follow_edgetype_cnt,
            item_follow_hour_cnt ,
            item_follow_day_cnt,
            item_follow_edgetype_hour_cnt ,
            item_follow_edgetype_day_cnt,
            hour,
            day,
            *user_hour_dist,
            *user_day_dist
        ]

        wt.write('{0},{1}\n'.format(words[0], ','.join([str(a) for a in features])))
                      

    rd.close()
    wt.close()
 

if __name__ == '__main__':
    # history_file = '/home/jialia/wsdm/edges_train_B.csv'
    # outpath = '/home/jialia/wsdm/seq_datasets/B_feature' 
    history_file = sys.argv[1]
    inpath = sys.argv[2]
    outpath = sys.argv[3]

    if not os.path.exists(outpath):
        os.mkdir(outpath)
    user2history, item2history, edge_type2freq = load_user_order_history(history_file)

     
    generate_features(
        os.path.join(inpath, 'train_instances.txt'),
        os.path.join(outpath, 'my_train.csv'),
        user2history,
        item2history
    )
    generate_features(
        os.path.join(inpath, 'valid_instances.txt'),
        os.path.join(outpath, 'my_valid.csv'),
        user2history,
        item2history
    )
    generate_features(
        os.path.join(inpath, 'valid.tsv'),
        os.path.join(outpath, 'valid.csv'),
        user2history,
        item2history
    )
    generate_features(
        os.path.join(inpath, 'inter_test.tsv'),
        os.path.join(outpath, 'inter_test.csv'),
        user2history,
        item2history
    )
    generate_features(
        os.path.join(inpath, 'final_test.tsv'),
        os.path.join(outpath, 'final_test.csv'),
        user2history,
        item2history
    )
