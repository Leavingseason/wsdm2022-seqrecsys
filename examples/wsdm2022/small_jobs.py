from collections import defaultdict
import os
from gensim.models import Word2Vec
import math

def get_selected_columns(infile, outfile, cols=[0], sep=','):
    with open(infile, 'r') as rd, open(outfile, 'w') as wt:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split(sep)
            wt.write('{0}\n'.format(','.join([words[i] for i in cols])))

def load_user_order_history(infile):
    user2history = {}  
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
            if userid not in user2history:
                user2history[userid] = list()      
            user2history[userid].append(itemid)
              
    return user2history 

def output_dict_len(d, outfile):
    with open(outfile, 'w') as wt:
        for k,v in d.items():
            wt.write('{0},{1}\n'.format(k,len(v)))

def concate_files(infiles, outfile):
    with open(outfile, 'w') as wt:
        for infile in infiles:
            with open(infile, 'r') as rd:
                while True:
                    line = rd.readline()
                    if not line:
                        break
                    wt.write(line)
        
def load_user_order_history(infile):
    user2history = {}
    item_set = set()
    edge_set = set()
    item2freq = defaultdict(int)
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
            item2freq[itemid] += 1
            edge_type = int(words[2])
            edge_set.add(edge_type)
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
    return user2history, item_set, edge_set, item2freq

def sentence_generation(infile, outfile):
    if not os.path.exists(os.path.dirname(outfile)):
        os.mkdir(os.path.dirname(outfile))
    user2history, item_set, edge_set, item2freq = load_user_order_history(infile)
    with open(outfile, 'w') as wt:
        for user, itemlist in user2history.items():
            wt.write('{0}\n'.format(' '.join([str(a[0]) for a in itemlist])))

def train_word2vec(infile, outfile):
    sentences = []
    with open(infile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split(' ')
            sentences.append(words)
    print('init word2vec')
    model = Word2Vec(sentences, min_count=1, size=32)
    # print('training word2vec')
    # model.train(total_examples=len(sentences), epochs=5)
    model.wv.save_word2vec_format(outfile, binary=False)

def sigmoid(x):
    return 1 / (1 + math.exp(-10*x))
def sigmoid_file(infile, outfile):
    with open(infile, 'r') as rd, open(outfile, 'w') as wt:
        while True:
            line = rd.readline()
            if not line:
                break
            s = float(line[:-1])
            wt.write('{0}\n'.format(sigmoid(s)))

if __name__ == '__main__':
    # get_selected_columns(
    #     '/home/jialia/wsdm/seq_datasets/A_feature_v4_max200_min1000_seq10_neg9/inter_test.csv',
    #     '/home/jialia/wsdm/seq_datasets/A_feature_v4_max200_min1000_seq10_neg9/inter_test_first2cols.csv',
    #     [0, 1],
    #     ','
    # )

    # history_file = '/home/jialia/wsdm/edges_train_A.csv' 
    # user2history  = load_user_order_history(history_file)
    # output_dict_len(user2history, '/home/jialia/wsdm/seq_datasets/output/user2itemlist_len_A.csv')

    # infiles = [
    #     '/home/jialia/wsdm/seq_datasets/A_edge_seqmerge_neg9/train_instances_1w.txt',
    #     '/home/jialia/wsdm/seq_datasets/A_edge_seqmerge_neg9/train_instances.txt'
    # ]
    # outfile = '/home/jialia/wsdm/seq_datasets/A_edge_seqmerge_neg9/train_instances_merged.txt'
    # concate_files(infiles, outfile)

    # sentence_generation(
    #     '/home/jialia/wsdm/edges_train_B.csv',
    #     '/home/jialia/wsdm/seq_datasets/word2vec/sentences_B.txt'
    # )
    # train_word2vec(
    #     '/home/jialia/wsdm/seq_datasets/word2vec/sentences_B.txt',
    #     '/home/jialia/wsdm/seq_datasets/word2vec/embeddings2_B.txt'
    #     )

    concate_files(
        [
            '/home/jialia/wsdm/seq_datasets/A_edge_300k_neg9_nocollide/train_instances.txt',
            '/home/jialia/wsdm/seq_datasets/A_edge_300k_neg9_nocollide/valid_instances.txt',
            '/home/jialia/wsdm/seq_datasets/A_edge_300k_neg9_nocollide_part1/train_instances.txt',
            '/home/jialia/wsdm/seq_datasets/A_edge_300k_neg9_nocollide_part1/valid_instances.txt',
            '/home/jialia/wsdm/seq_datasets/A_edge_300k_neg9_nocollide_part2/train_instances.txt'
        ],
        '/home/jialia/wsdm/seq_datasets/A_edge_300k_neg9_nocollide_merged/train_instances.txt'
    )