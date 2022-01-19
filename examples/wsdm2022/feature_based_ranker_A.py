import numpy as np
import networkx as nx
import random
import sys
from tqdm import * 
import os
import pickle

from recommenders.models.deeprec.deeprec_utils import cal_metric
 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import _tree
from sklearn import preprocessing

class FeatureBased(object):
    def __init__(self, seed = 10):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed) 
        self.coe = None
        self.model = None
        self.normalized_feature = True
    
    def fit_file(self, trainfile):
        train_features, train_labels = self.load_instances(trainfile)
        print('#. train features is {0}'.format(len(train_features)))
        if self.normalized_feature:
            self.feat_norm = preprocessing.StandardScaler().fit(train_features)
        self.fit(train_features=self.feat_norm.transform(train_features) if self.normalized_feature else train_features, train_labels=train_labels)
    
    def eval_file(self, testfile):
        features, labels = self.load_instances(testfile)
        res = self.eval(self.feat_norm.transform(features) if self.normalized_feature else features, labels)
        print(res)
        return res
    
    def infer_file(self, testfile, outfile ):
        test_features, _  = self.load_instances(testfile)
        print('infer scores of {0}...'.format(os.path.basename(testfile)))
        with open(outfile, 'w') as wt:
            scores = self.model.predict_proba(self.feat_norm.transform(test_features) if self.normalized_feature else test_features)
            for v in scores:
                wt.write('{0}\n'.format(v[1]))

    def get_file_line_cnt(self, file):
        cnt = 0
        with open(file, 'r') as rd:            
            while True:
                line = rd.readline()
                if not line:
                    break
                cnt += 1
                if cnt % 10000 == 0:
                    print('\rloading file {0} line {1}'.format(os.path.basename(file), cnt), end=' ')
        return cnt      
 

    def load_instances(self, file):
        file_line_cnt = self.get_file_line_cnt(file)
        print()
        features, labels = [], []
        with open(file, 'r') as rd:
            # for _ in tqdm(range(file_line_cnt), desc='loading {0}'.format(os.path.basename(file)), position=0, leave=True):
            for i in range(file_line_cnt):
                if i % 10000 == 0:
                    print('\r processing line {0} / {1}'.format(i, file_line_cnt), end=' ')
                line = rd.readline() 
                words = line[:-1].split(',')
                labels.append(int(words[0]))
                cur_instance = []
                for word in words[1:]:
                    cur_instance.append(float(word))  
                features.append(np.asarray(cur_instance, dtype=np.float32))
        return features, labels
    
    def fit(self, train_features = [], train_labels = []):
        print('Training Features Based algorithm...') 
        print('number of instances={0}'.format(len(train_features)))
        
        # model = LogisticRegression(solver='sag', random_state=10, C=10, max_iter=50)
        # model = DecisionTreeClassifier(max_depth=10)  #5 0.568664 

        ### A_feature_v4_max200_min1000_seq10_neg9_newfeature 0.5784
        # model = MLPClassifier(solver='adam', learning_rate_init=0.001, alpha=1e-7, hidden_layer_sizes=(16,), random_state=1, max_iter=50) #alpha=1e-5 :  0.580038   1e-6:0.5822  1e-7: 0.5833

        ###  A_feature_v4_max200_min1000_seq10_neg9_newfeature  0.580197
        ### 0.585648  A_feature_v4_max200_min1000_seq10_neg9   
        model = MLPClassifier(solver='adam', learning_rate_init=0.001, alpha=1e-7, hidden_layer_sizes=(32,16), random_state=1, max_iter=100) 
        
        
        model.fit(train_features, train_labels)
        
        # self.coe = list(model.coef_[0])
        # print('coe: {0}'.format(self.coe))
        
        self.model = model
        

    def eval(self, test_features = [], test_labels = []):
        print('evalation starts...')  
        print('number of instances: {0}'.format(len(test_features))) 

        scores = self.model.predict_proba(test_features)
        # print('eval : {0}'.format(scores[:10]))
        preds_rs = [v[1] + random.random() * 1e-6 for v in scores] 
        # preds_rs = np.array(preds_rs, dtype=np.float64)
             
        print('calculating metrics...')
        res = cal_metric(labels = np.array(test_labels, dtype=np.int32), preds = preds_rs, metrics = ["auc"])
        print('Feature Based algorithm done')
        return res
    
    def save_model(self, outfile):
        with open(outfile, 'wb') as wt:
            pickle.dump(self.model, wt)
    
    def load_model(self, infile):
        with open(infile, 'rb') as rd:
            self.model = pickle.load(rd)
 

if __name__ == '__main__':     
    inpath = '/home/jialia/wsdm/seq_datasets/A_feature_v4_max200_min1000_seq10_neg9'  ##A_feature_v4_max200_min1000_seq10_neg9  A_edge_seqmerge_neg9_test
    train_file = os.path.join(inpath, 'my_train.csv')
    valid_file = os.path.join(inpath, 'my_valid.csv')
    test_file = os.path.join(inpath, 'valid.csv')
    inter_test_file = os.path.join(inpath, 'inter_test.csv')
    final_test_file = os.path.join(inpath, 'final_test.csv')

    outpath = os.path.join(inpath, 'output')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
 
    model = FeatureBased()
    model.fit_file(train_file)
    res = model.eval_file(valid_file)
    print('evaluation on my valid file is {0}'.format(res))

    res = model.eval_file(test_file)
    print('evaluation on valid file is {0}'.format(res))
 
    if hasattr(model.model, 'tree_'):
        feat_importance = model.model.tree_.compute_feature_importances(normalize=False)
        print("feat importance = ")
        for i, v in enumerate(feat_importance):
            print('{0}\t{1}'.format(i, v))

    
    
    model.infer_file(
        inter_test_file,
        os.path.join(outpath, 'inter_test_output.txt')
        )
    model.infer_file(
        final_test_file,
        os.path.join(outpath, 'final_test_output.txt')
        ) 
    
    # model02 = FeatureBased()
    # model02.load_model(os.path.join(args['output_path'], 'feature-based-ranker{0}.pkl'.format(flag)))
    # print('model02')
    # model02.eval_file(os.path.join(args['data_path'], 'test.feature'), 100)
    # export_code(model02.model, os.path.join(args['output_path'], 'feature-based-ranker{0}-exported.py'.format(flag)), head_indent_num=2)

    print('Job: A_feature_v4_max200_min1000_seq10_neg9')

    ## A_edge_seqmerge_neg9_test:  0.579693
    ## A_feature_v4_max200_min1000_seq10_neg9_newfeature  0.580197
    ## A_feature_edge_seqmerge_neg9_vip5k_newfeature 0.582098 