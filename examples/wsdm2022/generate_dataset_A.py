from collections import defaultdict
import os
import random
from unicodedata import normalize 
from tqdm import tqdm
import numpy as np
import sys

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
        instance_type = 'context'
        while fail_cnt > 0:
            sampled_item, sampled_edgetype, sampled_time = itemid, edge_type, timestamp
            if random.random() < 0.5:
                sampled_edgetype = np.random.choice(edge_type2freq[0], 1, p=edge_type2freq[1])[0]
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
            for element in true_history:
                if collide(element, sampled_item, sampled_edgetype, sampled_time, 120):
                    flag = False
                    break
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
            for element in true_history:
                if collide(element, sampled_item, edge_type, timestamp, 120):
                    flag = False
                    break
            if flag:
                selected_item = sampled_item
                instance_type = 'replace_with_old_item'
                break
            else:
                fail_cnt -= 1

    if not selected_item:
        selected_item = random.sample(item_list, 1)[0]
    wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(0, userid, selected_item, selected_edgetype, selected_time, seq_items, seq_types, seq_times))
    return instance_type



def concat_elements(t, length, idx, max_hist_len):

    if length <=0:
        return '-1'
    if length > max_hist_len:
        start = length - max_hist_len
    else:
        start = 0
    res = ','.join([str(e[idx]) for e in t[start:length]])
    return res


def generate_test_set(user2history, infile, outfile, is_valid=True, max_hist_len=200):
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
            if userid in user2history:
                history = user2history[userid]
            else:
                history = {}
            seq_items, seq_types, seq_times = concat_elements(history, len(history), 0, max_hist_len), concat_elements(history, len(history), 1, max_hist_len), concat_elements(history, len(history), 2, max_hist_len)
            wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(label, userid, itemid, edge_type, int((timestamp01+timestamp02)/2), seq_items, seq_types, seq_times))
            
def normalize_dict_for_sample_weights(edge_type2freq):
    s = 0
    p = 0.75
    for v in edge_type2freq.values():
        s += pow(v, p)
    keys, values = [], []
    for k, v in edge_type2freq.items():
        keys.append(k)
        values.append(pow(v, p)/s)
    # return (keys, values)
    return (np.asarray(keys, dtype=np.int), np.asarray(values, dtype=np.float))



def generate_training_set(user2history, item_set, edge_type2freq, max_num_per_user, min_user_sequence, neg_num, train_ratio, outpath, allow_action, p_old_item, max_hist_len, p_context_corrupt=0.2):
    edge_type2freq = normalize_dict_for_sample_weights(edge_type2freq)
    item_list = list(item_set)
    user_list = list(user2history.keys())
    random.shuffle(user_list)
    # user_list = user_list[:100]
    valid_user_cnt = 0 
    instance_type2freq = defaultdict(int)
    vip_cnt = 10
    with open(os.path.join(outpath, 'train_instances.txt'), 'w') as wt01, open(os.path.join(outpath, 'valid_instances.txt'), 'w') as wt02:
        for userid in tqdm(user_list, desc='generating instances'):
            history = user2history[userid]
            if random.random() < train_ratio:
                wt = wt01
            else:
                wt = wt02
            cnt_history = len(history)
            if cnt_history <= min_user_sequence:
                if cnt_history <= 10:
                    continue
                if random.random() > 0.95:
                    continue
            valid_user_cnt += 1
            action_num = min(max_num_per_user, cnt_history - min_user_sequence) 
            action_num = max(action_num, 1)
            if cnt_history > 500000:
                if random.random() < 0.3 and vip_cnt > 0:
                    action_num = 2000
                    vip_cnt -= 1
            pos_idx_list = list(range(max(8, min(int(0.9 * cnt_history), cnt_history - action_num)), cnt_history))
            pos_idx_list = random.sample(pos_idx_list, action_num)
            pos_idx_list.sort()
            for pos_idx in pos_idx_list:  
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
                seq_items, seq_types, seq_times = concat_elements(history_proxy, pos_idx, 0, max_hist_len), concat_elements(history_proxy, pos_idx, 1, max_hist_len), concat_elements(history_proxy, pos_idx, 2, max_hist_len)
                wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(1, userid, action[0], action[1], action[2], seq_items, seq_types, seq_times))
                for _ in range(neg_num):
                    instance_type = gen_one_negative_instance(wt, userid, history_proxy, history, item_list, edge_type2freq, pos_idx, seq_items, seq_types, seq_times, p_old_item, p_context_corrupt)
                    instance_type2freq[instance_type] += 1
    print('#INFO: valid user cnt is {0} / {1}'.format(valid_user_cnt, len(user_list)))
    print('instance type 2 freq: ')
    print(instance_type2freq)


if __name__ == '__main__':
    # raw_data_path = '/home/jialia/wsdm' 
    # outpath = '/home/jialia/wsdm/seq_datasets/A_demo'

    raw_data_path = sys.argv[1]
    outpath = sys.argv[2]

    history_file = os.path.join(raw_data_path, 'edges_train_A.csv')
    
    ### set to a smaller number for quick demo
    ### if you want to reproduce a good result, do forget to replace the parameters with the commented numbers
    seq_len = 2  ##10
    max_per_user = 1 ## 200
    min_len = 1000
    neg_num = 1 ## 9
    
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    user2history, item_set, edge_type2freq = load_user_order_history(history_file)

    generate_training_set(user2history, item_set, edge_type2freq, max_per_user, min_len, neg_num, 0.8, outpath, allow_action=None, p_old_item=0.8, max_hist_len=seq_len, p_context_corrupt=0.4)
    generate_test_set(user2history,
        os.path.join(raw_data_path, 'input_A_initial.csv'),
        os.path.join(outpath, 'valid.tsv'),
        True,
        max_hist_len=seq_len
    )
    generate_test_set(user2history,
        os.path.join(raw_data_path, 'input_A.csv'),
        os.path.join(outpath, 'inter_test.tsv'),
        False,
        max_hist_len=seq_len
    )
    generate_test_set(user2history,
        os.path.join(raw_data_path, 'final', 'input_A.csv'),
        os.path.join(outpath,'final_test.tsv'),
        False,
        max_hist_len=seq_len
    )
