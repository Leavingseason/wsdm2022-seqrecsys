import sys
import os 
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)
from recommenders.datasets.amazon_reviews import download_and_extract, data_preprocessing, _create_vocab
from recommenders.datasets.download_utils import maybe_download


from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel 
# from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel as SeqModel
# from recommenders.models.deeprec.models.sequential.caser import CaserModel as SeqModel
# from recommenders.models.deeprec.models.sequential.gru4rec import GRU4RecModel as SeqModel
# from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel

#from recommenders.models.deeprec.models.sequential.nextitnet import NextItNetModel

from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
#from recommenders.models.deeprec.io.nextitnet_iterator import NextItNetIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

yaml_file = '/home/jialia/wsdm/src/recommenders/examples/wsdm2022/sli_rec_B.yaml'  
  
RANDOM_SEED = SEED  # Set None for non-deterministic result

# data_path = os.path.join("tests", "resources", "deeprec", "slirec")
# data_path = '/home/jialia/wsdm/seq_datasets/B_full_feature_v2'
data_path = sys.argv[1]
print(os.path.abspath(data_path))  ## the path where I enter the cmd

# for test
train_file = os.path.join(data_path, r'train_instances.txt')
valid_file = os.path.join(data_path, r'valid_instances.txt')
test_file = os.path.join(data_path, r'valid.tsv')
pred_file = os.path.join(data_path, r'inter_test.tsv')
final_pred_file = os.path.join(data_path, r'final_test.tsv')
user_vocab = os.path.join(data_path, r'user_vocab.pkl')
item_vocab = os.path.join(data_path, r'item_vocab.pkl')
cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
output_file = os.path.join(data_path, r'inter_test_output.txt')
submit_file = os.path.join(data_path, r'final_test_output.txt')


train_num_ngs = 9 # number of negative instances with a positive instance for training
valid_num_ngs = 9 # number of negative instances with a positive instance for validation
test_num_ngs = 9 # number of negative instances with a positive instance for testing


_create_vocab(
    [train_file, valid_file],
    user_vocab, item_vocab, cate_vocab
)


### NOTE:  
### remember to use `_create_vocab(train_file, user_vocab, item_vocab, cate_vocab)` to generate the user_vocab, item_vocab and cate_vocab files, if you are using your own dataset rather than using our demo Amazon dataset.
hparams = prepare_hparams(yaml_file, 
                        #   user_dropout=False,
                          embed_l2=0.,  
                          layer_l2=0., 
                          enable_BN=True, ##-- True
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable  else 0.001
                          epochs=100000,
                          EARLY_STOP=40000,
                          batch_size=400,
                          show_step=5000,
                          MODEL_DIR=os.path.join(data_path, "model/"),
                          SUMMARIES_DIR=os.path.join(data_path, "summary/"),
                          user_vocab=user_vocab,
                          item_vocab=item_vocab,
                          cate_vocab=cate_vocab,
                          need_sample=False,
                          train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                          loss='log_loss', #'log_loss', 'softmax'
                          max_seq_length=50,
                          cont_feat_len=85,
                          use_cont_feat=False,
                          init_item_emb=False, 
                          shuffle=True
            )
print(hparams.values)

input_creator = SequentialIterator


model = SeqModel(hparams, input_creator, seed=RANDOM_SEED)

# model.load_model(os.path.join(data_path, "model_20220118_20k_0.8923", 'step_20000'))

with Timer() as train_time:
    model = model.fit(train_file, valid_file, valid_num_ngs=9, eval_metric='auc')
print('Time cost for training is {0:.2f} mins'.format(train_time.interval/60.0))

### model = model.fit(test_file, test_file, valid_num_ngs=9, eval_metric='auc') ##-- quick test

model.load_model(os.path.join(data_path, "model", 'best_model'))


res_syn = model.run_eval(test_file, num_ngs=9)
print(res_syn)
model.predict(pred_file, output_file)
model.predict(final_pred_file, submit_file)
# print('Job finished. B, continue training = 20k, seq=50')
# print('Job finished. B_v2, epoch=50k, seq=100')
## ASVD: 0.867497
## GRU:  0.877529
## SLi-Rec: 0.892736 
## B_v4: 0.8937

print("Job:B_full_feature_v2, with BN, no cont feat, seq=50, shuffle=True")

## B_full_feature_v2 no cont_feat, with BN
##5k: 0.8778
##10k: 0.8827
##20k: 0.8848
##25k: 0.8824
##35k: 0.8878
##40k: 0.8903
##45k: 0.8876
##50k: 0.8925
##55k: 0.8903
##60k: 0.8894
##65k: 0.8904
##70k: 0.8814
##75k: 0.8896
##80k: 0.8871
##85k: 0.8920

## with shuffle:
##5k: 0.8793
##10k: 0.8884
##15k: 0.8898
##20k: 0.8923
##25k: 0.8908
##30k: 0.8895
##35k: 0.8888
##40k: 0.8913
##45k: 0.8909
##50k: 0.8876
##65k: 0.8881