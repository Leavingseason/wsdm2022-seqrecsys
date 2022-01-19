import os
import numpy as np
from sklearn import preprocessing


def job_append_cont_features():
    seq_path = '/home/jialia/wsdm/seq_datasets/A_v4_max200_min1000_seq10_neg9'
    seq_files = [
        'train_instances.txt',
        'valid_instances.txt',
        'valid.tsv',
        'inter_test.tsv',
        'final_test.tsv'
    ]

    feat_path = '/home/jialia/wsdm/seq_datasets/A_feature_v4_max200_min1000_seq10_neg9'
    feat_files = [
        'my_train.csv',
        'my_valid.csv',
        'valid.csv',
        'inter_test.csv',
        'final_test.csv'
    ]

    out_path = '/home/jialia/wsdm/seq_datasets/A_full_v4_max200_min1000_seq10_neg9' 

    _process(
        seq_path, seq_files,
        feat_path, feat_files,
        out_path
    )

def job_append_cont_features_B():
    ## v2: 85 col
    
    seq_path = '/home/jialia/wsdm/seq_datasets/B_v4'
    seq_files = [
        'train_instances.txt',
        'valid_instances.txt',
        'valid.tsv',
        'inter_test.tsv',
        'final_test.tsv'
    ]

    feat_path = '/home/jialia/wsdm/seq_datasets/B_feature_v4'
    feat_files = [
        'my_train.csv',
        'my_valid.csv',
        'valid.csv',
        'inter_test.csv',
        'final_test.csv'
    ]

    out_path = '/home/jialia/wsdm/seq_datasets/B_full_feature_v4' 

    _process(
        seq_path, seq_files,
        feat_path, feat_files,
        out_path
    )

def _process(
        seq_path, seq_files,
        feat_path, feat_files,
        out_path
    ):
    assert out_path != feat_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    col_n = 0
    feat_trans = None
    for i in range(len(seq_files)):
        print(i)
        seq_file = os.path.join(seq_path, seq_files[i])
        feat_file = os.path.join(feat_path, feat_files[i])
        outfile = os.path.join(out_path, seq_files[i])

        feats = []
        with open(feat_file, 'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                words = line[:-1].split(',')
                row = [float(a) for a in words[1:]]
                col_n = len(row)
                feats.append(np.array(row, dtype=np.float32))
        
        if not feat_trans:
            feat_trans = preprocessing.StandardScaler().fit(feats)
        feats = feat_trans.transform(feats)

        with open(seq_file, 'r') as rd, open(outfile, 'w') as wt:
            idx = 0
            while True:
                line = rd.readline()
                if not line:
                    break
                cont_feat_str = ','.join([str(a) for a in feats[idx]])
                idx += 1 
                wt.write('{0}\t{1}\n'.format(line[:-1], cont_feat_str))
        assert idx == len(feats)
    print('#. columns is {0}'.format(col_n))

if __name__ == '__main__':
    # job_append_cont_features()

    job_append_cont_features_B()