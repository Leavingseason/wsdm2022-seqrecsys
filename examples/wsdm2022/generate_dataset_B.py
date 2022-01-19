from collections import defaultdict
import os
import random
import math
from tqdm import tqdm 

def load_user_order_history(infile):
    user2history = {}
    item_set = set()
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
            if userid not in user2history:
                user2history[userid] = []
            user2history[userid].append((itemid, edge_type, timestamp))
    print('sorting...', end='\t')
    for k,v in user2history.items():
        user2history[k].sort(key=lambda t:t[2])
    print('done.')
    print('#.users = {0}'.format(len(user2history)))
    return user2history, item_set

def collide(element, itemid, edge_type, timestamp, tol=40):
    if element[0] == itemid and element[1] == edge_type and abs((element[2]-timestamp)/(60*60)) <= tol:
        return True
    else:
        return False

def gen_one_negative_instance(wt, userid, history, item_list, pos_idx, seq_items, seq_types, seq_times): 
    p_old_item = 0.2
    fail_cnt = 10
    selected_item = None 
    itemid, edge_type, timestamp= history[pos_idx][0],  history[pos_idx][1],  history[pos_idx][2]
    if random.random() < p_old_item: 
        valid_item_list = list(set([t[0] for t in history[:pos_idx]]))
        while fail_cnt > 0:
            sampled_item = random.sample(valid_item_list, 1)[0]
            flag = True
            for element in history:
                if collide(element, sampled_item, edge_type, timestamp):
                    flag = False
                    break
            if flag:
                selected_item = sampled_item
                break
            else:
                fail_cnt -= 1
    if not selected_item:
        selected_item = random.sample(item_list, 1)[0]
    wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(0, userid, selected_item, edge_type, timestamp, seq_items, seq_types, seq_times))



def concat_elements(t, length, idx):
    if length <=0:
        return ''
    res = ','.join([str(e[idx]) for e in t[0:length]])
    return res


def generate_test_set(user2history, infile, outfile, is_valid=True):
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
            seq_items, seq_types, seq_times = concat_elements(history, len(history), 0), concat_elements(history, len(history), 1), concat_elements(history, len(history), 2)
            wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(label, userid, itemid, edge_type, int((timestamp01+timestamp02)/2), seq_items, seq_types, seq_times))
            


def generate_training_set(user2history, item_set, max_num_per_user, min_user_sequence, neg_num, train_ratio, outpath, allow_action):
    item_list = list(item_set)
    user_list = list(user2history.keys())
    random.shuffle(user_list)
    valid_user_cnt = 0
    with open(os.path.join(outpath, 'train_instances.txt'), 'w') as wt01, open(os.path.join(outpath, 'valid_instances.txt'), 'w') as wt02:
        for userid  in tqdm(user_list, desc='gen training instances'):
            history = user2history[userid]
            if random.random() < train_ratio:
                wt = wt01
            else:
                wt = wt02
            cnt_history = len(history)
            if cnt_history <= min_user_sequence:
                continue
            valid_user_cnt += 1
            action_num = min(max_num_per_user, cnt_history - min_user_sequence)
            for i in range(action_num):
                pos_idx = cnt_history - (action_num-i)
                action = history[pos_idx]
                if action[1] not in allow_action:
                    if random.random() > 0.05:
                        continue
                seq_items, seq_types, seq_times = concat_elements(history, pos_idx, 0), concat_elements(history, pos_idx, 1), concat_elements(history, pos_idx, 2)
                wt.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(1, userid, action[0], action[1], action[2], seq_items, seq_types, seq_times))
                for _ in range(neg_num):
                    gen_one_negative_instance(wt, userid, history, item_list, pos_idx, seq_items, seq_types, seq_times)


if __name__ == '__main__':
    history_file = '/home/jialia/wsdm/edges_train_B.csv'
    outpath = '/home/jialia/wsdm/seq_datasets/B_final'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    user2history, item_set = load_user_order_history(history_file)
    generate_training_set(user2history, item_set, 10, 5, 9, 0.8, outpath, allow_action=set([12]))
    generate_test_set(user2history,
        '/home/jialia/wsdm/input_B_initial.csv',
        os.path.join(outpath, 'valid.tsv'),
        True
    )
    generate_test_set(user2history,
        '/home/jialia/wsdm/input_B.csv',
        os.path.join(outpath, 'inter_test.tsv'),
        False
    )

    generate_test_set(user2history,
        '/home/jialia/wsdm/final/input_B.csv',
        os.path.join(outpath, 'final_test.tsv'),
        False
    )