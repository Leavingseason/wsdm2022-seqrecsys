import sys
import os 
from tempfile import TemporaryDirectory
from tkinter import TRUE
import numpy as np
import tensorflow.compat.v1 as tf
import shutil
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)
from recommenders.datasets.amazon_reviews import download_and_extract, data_preprocessing, _create_vocab
from recommenders.datasets.download_utils import maybe_download


from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
####  to use the other model, use one of the following lines:
# from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel as SeqModel
# from recommenders.models.deeprec.models.sequential.caser import CaserModel as SeqModel
# from recommenders.models.deeprec.models.sequential.gru4rec import GRU4RecModel as SeqModel
# from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel

#from recommenders.models.deeprec.models.sequential.nextitnet import NextItNetModel

from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
#from recommenders.models.deeprec.io.nextitnet_iterator import NextItNetIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

yaml_file = '/home/jialia/wsdm/src/recommenders/examples/wsdm2022/sli_rec_A.yaml'  
EPOCHS = 6
BATCH_SIZE = 400
RANDOM_SEED = SEED  # Set None for non-deterministic result

# data_path = os.path.join("tests", "resources", "deeprec", "slirec")
# data_path = '/home/jialia/wsdm/seq_datasets/A_full_v4_max200_min1000_seq10_neg9'
data_path = sys.argv[1]
print(os.path.abspath(data_path))  ## the path where I enter the cmd

model_path = os.path.join(data_path, "model")
summary_path = os.path.join(data_path, "summary")
# try:
#     shutil.rmtree(model_path)
#     shutil.rmtree(summary_path)
# except OSError as e:
#     print("Error: %s - %s." % (e.filename, e.strerror))

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

reviews_name = 'reviews_Movies_and_TV_5.json'
meta_name = 'meta_Movies_and_TV.json'
reviews_file = os.path.join(data_path, reviews_name)
meta_file = os.path.join(data_path, meta_name)
train_num_ngs = 9 # number of negative instances with a positive instance for training
valid_num_ngs = 9 # number of negative instances with a positive instance for validation
test_num_ngs = 9 # number of negative instances with a positive instance for testing
sample_rate = 0.01 # sample a small item set for training and testing here for fast example

input_files = [reviews_file, meta_file, train_file, valid_file, test_file, user_vocab, item_vocab, cate_vocab]


# if not os.path.exists(user_vocab):
_create_vocab(
    [train_file, valid_file],
    user_vocab, item_vocab, cate_vocab
)


# hparams = prepare_hparams(yaml_file, 
#                           embed_l2=0., 
#                           layer_l2=0., 
#                           learning_rate=0.001,  # 0.001
#                           epochs=EPOCHS,
#                           batch_size=BATCH_SIZE,
#                           show_step=1000,
#                           MODEL_DIR=model_path,
#                           SUMMARIES_DIR=summary_path,
#                           user_vocab=user_vocab,
#                           item_vocab=item_vocab,
#                           cate_vocab=cate_vocab,
#                           need_sample=False,
#                           train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
#                           loss='log_loss', #'log_loss', 'softmax',
#                           max_seq_length=500
#             )
hparams = prepare_hparams(yaml_file, 
                          user_dropout=False,
                          embed_l2=0.,  
                          layer_l2=0., 
                          enable_BN=False,
                          learning_rate=0.001,  # set to 0.01 if batch normalization is disable else 0.001
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
                          max_seq_length=10,
                          cont_feat_len=85, ##78 or 85,
                          use_cont_feat=True,
                          use_useritem_type=0,
                          init_item_emb=False
            )
print(hparams.values)

input_creator = SequentialIterator


model = SeqModel(hparams, input_creator, seed=RANDOM_SEED)

# model.load_model(os.path.join(data_path, "model", 'step_50000')) 

with Timer() as train_time:
    model = model.fit(train_file, valid_file, valid_num_ngs=9, eval_metric='auc')  ##--
print('Time cost for training is {0:.2f} mins'.format(train_time.interval/60.0))
 
model.load_model(os.path.join(data_path, "model", 'best_model'))


res_syn = model.run_eval(test_file, num_ngs=9)
print(res_syn)

model.predict(pred_file, output_file)
model.predict(final_pred_file, submit_file)
print('SLI-Rec A_full_v4_max200_min1000_seq10_neg9, cont feat type = 2')

##SLI-Rec A_full_v4_max200_min1000_seq10_neg9, only cont feat
##10k: 0.5499
##20k: 0.5628
##30k: 0.5799
##45k: 0.5757
##50k: 0.5484
##60k: 0.5756
##70k: 0.5727


## SLI-Rec A_full_v4_max200_min1000_seq10_neg9, cont feat type = 2
##20k