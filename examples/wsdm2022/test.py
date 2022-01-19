import random
from datetime import datetime
from collections import defaultdict
import os

def load_user_history(infile, add_reverse=False):
    user2item2type = {}
    itemset = set()
    edgetype2freq = defaultdict(int)
    cnt = 0
    with open(infile, 'r') as rd:
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
            itemset.add(itemid)
            edge_type = int(words[2])
            edgetype2freq[edge_type] += 1
            if userid not in user2item2type:
                user2item2type[userid] = defaultdict(set)
            user2item2type[userid][itemid].add(edge_type)
            if add_reverse:
                if itemid not in user2item2type:
                    user2item2type[itemid] = defaultdict(set)
                user2item2type[itemid][userid].add(edge_type)
    print('#.users is {0}, #.items is {1}'.format(len(user2item2type), len(itemset)))
    print('#.overlap of userid and itemid is {0}'.format(len(set(user2item2type.keys())&itemset)))
    return user2item2type, edgetype2freq

def output_dict(outfile, d, idx):
    d_list = [(k,v) for k, v in d.items()]
    d_list.sort(reverse=True, key=lambda t:t[idx])
    with open(outfile, 'w') as wt:
        for (k,v) in d_list:
            wt.write('{0}\t{1}\n'.format(k,v))

def transition_test(train_file, test_file, outpath, flag='A', simplified=False):
    user2item2type, edgetype2freq = load_user_history(train_file, False)
    type2freq_pos = defaultdict(int)
    type2freq_neg = defaultdict(int)
    with open(test_file, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break 
            words = line[:-1].split(',')
            userid = int(words[0])
            itemid = int(words[1])  
            edge_type = int(words[2])
            label = int(words[5])
            if userid not in user2item2type or itemid not in user2item2type[userid]:
                type_key = 'none=>{0}'.format(edge_type)
            elif simplified:
                type_key = 'exist=>{0}'.format(edge_type)
            else:
                actions = list(user2item2type[userid][itemid])
                actions.sort()
                type_key = ','.join([str(a) for a in actions])
                type_key = '{0}=>{1}'.format(type_key,edge_type)
            if label > 0:
                type2freq_pos[type_key] += 1
            else:
                type2freq_neg[type_key] += 1
    output_dict(os.path.join(outpath, 'transition_pos_{0}.tsv'.format(flag)), type2freq_pos, 1)        
    output_dict(os.path.join(outpath, 'transition_neg_{0}.tsv'.format(flag)), type2freq_neg, 1)        
    output_dict(os.path.join(outpath, 'edge_type_freq_{0}.tsv'.format(flag)), edgetype2freq, 1)  
    


def time_interval_test(infile, outfile):
    hour2freq = defaultdict(int)
    with open(infile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.split(',')
            start = datetime.fromtimestamp(int(words[3]))
            end = datetime.fromtimestamp(int(words[4]))
            hours = int((end-start).total_seconds()/(60*60))
            hour2freq[hours] += 1
    hours_freq = [(k,v) for k,v in hour2freq.items()]
    hours_freq.sort(key=lambda t:t[0])
    with open(outfile, 'w') as wt:
        for a in hours_freq:
            wt.write('{0},{1}\n'.format(a[0],a[1]))

def fake_sub(infile, outfile):
    with open(infile, 'r') as rd:
        lines = rd.readlines()
    with open(outfile, 'w') as wt:
        for _ in lines:
            wt.write('{0}\n'.format(random.random()))

if __name__ == '__main__':
    # fake_sub('/home/jialia/wsdm/input_A.csv', '/home/jialia/wsdm/output_A.csv')
    # fake_sub('/home/jialia/wsdm/input_B.csv', '/home/jialia/wsdm/output_B.csv')
    # time_interval_test('/home/jialia/wsdm/final/input_B.csv','/home/jialia/wsdm/data_pattern/final_input_B.csv')
    # transition_test(
    #     '/home/jialia/wsdm/edges_train_A.csv',
    #     '/home/jialia/wsdm/input_A_initial.csv',
    #     '/home/jialia/wsdm/data_pattern',
    #     'A', True
    # )

    transition_test(
        '/home/jialia/wsdm/edges_train_B.csv',
        '/home/jialia/wsdm/input_B_initial.csv',
        '/home/jialia/wsdm/data_pattern',
        'B',
        False
    )