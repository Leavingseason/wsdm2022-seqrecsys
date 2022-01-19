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

class FeatureBased(object):
    def __init__(self, seed = 10):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed) 
        self.coe = None
        self.model = None
    
    def fit_file(self, trainfile):
        train_features, train_labels = self.load_instances(trainfile)
        self.fit(train_features=train_features, train_labels=train_labels)
    
    def eval_file(self, testfile):
        features, labels = self.load_instances(testfile)
        res = self.eval(features, labels)
        print(res)
        return res
    
    def infer_file(self, testfile, outfile ):
        test_features, _  = self.load_instances(testfile)
        print('infer scores starts...')
        with open(outfile, 'w') as wt:
            scores = self.model.predict_proba(test_features)
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
        return cnt    
               
    
    # def load_instances(self, file):
    #     file_line_cnt = self.get_file_line_cnt(file)
    #     features, labels = np.empty(file_line_cnt, dtype=object), np.zeros(file_line_cnt, dtype=np.int)
    #     with open(file, 'r') as rd:
    #         for cnt in tqdm(range(file_line_cnt), desc='loading {0}'.format(os.path.basename(file))):
    #             line = rd.readline() 
    #             words = line[:-1].split(',')
    #             labels[cnt]=int(words[0])
    #             cur_instance = []
    #             for word in words[1:]:
    #                 cur_instance.append(float(word))  
    #             features[cnt] = np.asarray(cur_instance, dtype=np.float32)  
    #     return features, labels

    def load_instances(self, file):
        file_line_cnt = self.get_file_line_cnt(file)
        print()
        features, labels = [], []
        with open(file, 'r') as rd:
            # for _ in tqdm(range(file_line_cnt), desc='loading {0}'.format(os.path.basename(file)), position=0, leave=True):
            for i in tqdm(range(file_line_cnt), desc='loading {0}'.format(os.path.basename(file))): 
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
        model = DecisionTreeClassifier(max_depth=5)  #5
        # model = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(32,), random_state=1)
        
        
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
    
    train_file = '/home/jialia/wsdm/seq_datasets/B_feature_v2/my_train.csv'
    valid_file = '/home/jialia/wsdm/seq_datasets/B_feature_v2/my_valid.csv'
    test_file = '/home/jialia/wsdm/seq_datasets/B_feature_v2/valid.csv'
    inter_test_file = '/home/jialia/wsdm/seq_datasets/B_feature_v2/inter_test.csv'
    final_test_file = '/home/jialia/wsdm/seq_datasets/B_feature_v2/final_test.csv'

    outpath = os.path.join('/home/jialia/wsdm/seq_datasets/B_feature_v2', 'output')
    if not os.path.exists(outpath):
        os.mkdir(outpath)
 
    model = FeatureBased()
    model.fit_file(train_file)
    res = model.eval_file(valid_file)
    print('evaluation on my valid file is {0}'.format(res))

    res = model.eval_file(test_file)
    print('evaluation on valid file is {0}'.format(res))
 
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