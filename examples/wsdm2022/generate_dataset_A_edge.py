from collections import defaultdict
import os
import random
from unicodedata import normalize 
from tqdm import tqdm
import numpy as np
import math

def load_user_order_history(infile):
    user2history = {}
    item_set = set()
    edge_type2freq = defaultdict(int)
    with open(infile, 'r') as rd:
        cnt = 0
        while True:
            line = rd.readline()
            if not line:
                break
            if cnt % 10000 == 0:
                print('\r loading line {0}'.format(cnt), end=' ')
            cnt += 1
            words = line[:-1].split(',')
            userid = int(words[0])
            itemid = int(words[1])
            edge_type = int(words[2])
            timestamp = int(words[3])
            item_set.add(itemid)
            edge_type2freq[edge_type] += 1
            if userid not in user2history:
                user2history[userid] = []
            user2history[userid].append((itemid, edge_type, timestamp))
    print('sorting...', end='\t')
    for k,v in user2history.items():
        user2history[k].sort(key=lambda t:t[2])
    print('done.')
    print('#.users = {0}'.format(len(user2history)))
    return user2history, item_set, edge_type2freq

def collide(element, itemid, edge_type, timestamp, tol=40):
    if element[0] == itemid and element[1] == edge_type and abs((element[2]-timestamp)/(60*60)) <= tol:
        return True
    else:
        return False

def gen_one_negative_instance(wt, userid, history, true_history, item_list, edge_type2freq, pos_idx, seq_items, seq_types, seq_times, p_old_item=0.8, p_context_corrupt=0.2):       
    instance_type = 'random_item'
    itemid, edge_type, timestamp = history[pos_idx][0],  history[pos_idx][1],  history[pos_idx][2]
    valid_item_list = list(set([t[0] for t in history[:pos_idx]]))

    selected_item, selected_edgetype, selected_time = None, edge_type, timestamp
    if random.random() < p_context_corrupt: 
        fail_cnt = 50 
        while fail_cnt > 0: 
            instance_type = 'context'
            sampled_item, sampled_edgetype, sampled_time = itemid, edge_type, timestamp
            if random.random() < 0.4:
                sampled_edgetype = random.sample(edge_type2freq, 1)[0] #np.random.choice(edge_type2freq[0], 1, p=edge_type2freq[1])[0]
                instance_type += '_edgetype'
            if random.random() < 0.3:
                sampled_item = random.sample(valid_item_list, 1)[0]
                instance_type += '_item'
            if random.random() < 0.2:
                instance_type += '_time'
                time_drift = random.randint(125, 500)
                if random.random() < 0.5:
                    time_drift *= -1
                sampled_time = timestamp + time_drift * 60 * 60

            flag = True
            # for element in true_history:
            #     if collide(element, sampled_item, sampled_edgetype, sampled_time, 120):
            #         flag = False
            #         break
            if instance_type == 'context':
                flag = False
            if flag:
                selected_item = sampled_item
                selected_edgetype = selected_edgetype
                selected_time = sampled_time 
                break
            else:
                fail_cnt -= 1

    if not selected_item and random.random() < p_old_item: 
        fail_cnt = 10
        
        while fail_cnt > 0:
            sampled_item = random.sample(valid_item_list, 1)[0]
            flag = True
            # for element in true_history:
            #     if collide(element, sampled_item, edge_type, timestamp, 120):
            #         flag = False
            #         break
            if flag:
                selected_item = sampled_item
                instance_type = 'replace_with_old_item'
                break
            else:
                fail_cnt -= 1

    if not selected_item:
        instance_type = 'random_item'
        selected_item = random.sample(item_list, 1)[0]
    wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(0, userid, selected_item, selected_edgetype, selected_time, seq_items, seq_types, seq_times))
    return instance_type

def convert_freq(v):
    t = int(math.log(v+1, 2))
    return 'n{0}'.format(t)
    

def concat_elements(t, length, idx, max_hist_len):

    if length <=0:
        return '-1'
    if length > max_hist_len:
        start = length - max_hist_len
    else:
        start = 0
    res = ','.join([str(e[idx]) for e in t[start:length]])
    return res


def merge_elements(t, end_idx, max_hist_len):
    if end_idx <=0:
        return []
 
    item2freq, item2lasttime = defaultdict(int), defaultdict(int)
    for item in t[:end_idx]:
        item2freq[item[0]] += 1
        if item2lasttime[item[0]] < item[2]:
            item2lasttime[item[0]] = item[2]
    item2list = [(a, convert_freq(item2freq[a]), item2lasttime[a]) for a in item2freq.keys()]
    item2list.sort(key=lambda t:t[2])

    N = len(item2list)
    
    if N > max_hist_len:
        start = N - max_hist_len
    else:
        start = 0 
    return item2list[start:]


def generate_test_set(user2history, infile, outfile, is_valid=True, max_hist_len=200):
    node2cache = {}
    with open(infile, 'r') as rd, open(outfile, 'w') as wt:
        cnt = 0
        while True:
            line = rd.readline()
            if not line:
                break
            if cnt % 10000 == 0:
                print('\r loading line {0}'.format(cnt), end=' ')
            cnt += 1
            words = line[:-1].split(',')
            userid = int(words[0])
            itemid = int(words[1])
            edge_type = int(words[2])
            timestamp01 = int(words[3])
            timestamp02 = int(words[4])
            if is_valid:
                label = int(words[5])
            else:
                label=0

            if userid not in node2cache:
                if userid in user2history:
                    history = user2history[userid]
                else:
                    history = {}  
                merged_history = merge_elements(history, len(history), max_hist_len)
                seq_items = ','.join([str(a[0]) for a in merged_history]) if len(merged_history) > 0 else "-1"
                seq_types = ','.join([str(a[1]) for a in merged_history]) if len(merged_history) > 0 else "-1"
                seq_times = ','.join([str(a[2]) for a in merged_history]) if len(merged_history) > 0 else "-1"
                node2cache[userid] = (seq_items, seq_types, seq_times)
            else:
                seq_items, seq_types, seq_times = node2cache[userid] 
            # seq_items, seq_types, seq_times = concat_elements(history, len(history), 0, max_hist_len), concat_elements(history, len(history), 1, max_hist_len), concat_elements(history, len(history), 2, max_hist_len)
            wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(label, userid, itemid, edge_type, int((timestamp01+timestamp02)/2), seq_items, seq_types, seq_times))
            
def normalize_dict_for_sample_weights(edge_type2freq, p=0.75):
    s = 0 
    for v in edge_type2freq.values():
        s += pow(v, p)
    keys, values = [], []
    for k, v in edge_type2freq.items():
        keys.append(k)
        values.append(pow(v, p)/s)
    # return (keys, values)
    return (np.asarray(keys, dtype=np.int), np.asarray(values, dtype=np.float))



def generate_training_set(edge_N, user2history, item_set, edge_type2freq, min_user_sequence, neg_num, train_ratio, outpath, allow_action, p_old_item, max_hist_len, p_context_corrupt=0.2):
    edge_type2freq = list(edge_type2freq.keys())
    item_list = list(item_set) 
    user2itemcnt = {k:len(v) for k,v in user2history.items()}
    user_array, user_prob = normalize_dict_for_sample_weights(user2itemcnt, p=1.0)
     
    valid_user_cnt = 0 
    instance_type2freq = defaultdict(int)
    vip_cnt = 10
    selected_user2freq = defaultdict(int) 
    with open(os.path.join(outpath, 'train_instances.txt'), 'w') as wt01, open(os.path.join(outpath, 'valid_instances.txt'), 'w') as wt02:
        for _ in tqdm(range(edge_N), desc='generating instances'):
            userid = np.random.choice(user_array, 1, p=user_prob)[0]
            history = user2history[userid]
            selected_user2freq[userid] += 1
            if random.random() < train_ratio:
                wt = wt01
            else:
                wt = wt02
            cnt_history = len(history)
            if cnt_history <= min_user_sequence: 
                continue
            valid_user_cnt += 1
            
            pos_idx = cnt_history - selected_user2freq[userid] 
            if pos_idx <= 5:
                continue

            if random.random() < 0.1 and pos_idx > 1000:
                skip_num = random.randint(0, pos_idx//3)
                history_proxy = history[0:(pos_idx-skip_num)] + history[pos_idx:] 
                pos_idx -= skip_num
            else:
                history_proxy = history

            action = history_proxy[pos_idx]
            if allow_action is not None and action[1] not in allow_action:
                if random.random() > 0.05:
                    continue
            merged_history = merge_elements(history_proxy, pos_idx, max_hist_len)
            seq_items = ','.join([str(a[0]) for a in merged_history]) if len(merged_history) > 0 else "-1"
            seq_types = ','.join([str(a[1]) for a in merged_history]) if len(merged_history) > 0 else "-1"
            seq_times = ','.join([str(a[2]) for a in merged_history]) if len(merged_history) > 0 else "-1"
            # seq_items, seq_types, seq_times = concat_elements(history_proxy, pos_idx, 0, max_hist_len), concat_elements(history_proxy, pos_idx, 1, max_hist_len), concat_elements(history_proxy, pos_idx, 2, max_hist_len)

            wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(1, userid, action[0], action[1], action[2], seq_items, seq_types, seq_times))
            for _ in range(neg_num):
                instance_type = gen_one_negative_instance(wt, userid, history_proxy, history, item_list, edge_type2freq, pos_idx, seq_items, seq_types, seq_times, p_old_item, p_context_corrupt)
                instance_type2freq[instance_type] += 1
    print('#INFO: valid user cnt is {0} / {1}'.format(valid_user_cnt, len(user_array)))
    print('instance type 2 freq: ')
    print(instance_type2freq)


if __name__ == '__main__':
    history_file = '/home/jialia/wsdm/edges_train_A.csv'
    outpath = '/home/jialia/wsdm/seq_datasets/A_edge_300k_neg9_nocollide_testset'
    seq_len = 100 
    min_len = 10
    neg_num = 9
    edge_N = 100000
    
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    user2history, item_set, edge_type2freq = load_user_order_history(history_file)

    # generate_training_set(edge_N, user2history, item_set, edge_type2freq, min_len, neg_num, 0.8, outpath, allow_action=None, p_old_item=0.8, max_hist_len=seq_len, p_context_corrupt=0.4)
    generate_test_set(user2history,
        '/home/jialia/wsdm/input_A_initial.csv',
        os.path.join(outpath, 'valid.tsv'),
        True,
        max_hist_len=seq_len
    )
    generate_test_set(user2history,
        '/home/jialia/wsdm/input_A.csv',
        os.path.join(outpath, 'inter_test.tsv'),
        False,
        max_hist_len=seq_len
    )
    generate_test_set(user2history,
        '/home/jialia/wsdm/final/input_A.csv',
        os.path.join(outpath,'final_test.tsv'),
        False,
        max_hist_len=seq_len
    )
