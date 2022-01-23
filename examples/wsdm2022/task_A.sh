#!/bin/bash

## source /anaconda/bin/activate && conda activate py37_tensorflow && bash task_A.sh

raw_data_path='/home/jialia/wsdm'  ## paste all the raw data files (except for the final test set) into this folder. Create a folder named 'final' and put the 'input_A.csv' and 'input_B.csv' of final test set into this folder.
output_path='/home/jialia/wsdm/seq_datasets'
output_tag='A_demo'

export PYTHONPATH=/home/jialia/wsdm/submit/recommenders
cd /home/jialia/wsdm/submit/recommenders/examples/wsdm2022


#### for quick test of the pipeline, data parameters are set to a small number. For full data, please note the comments in Line 223 of generate_dataset_A.py
python generate_dataset_A.py $raw_data_path $output_path'/'$output_tag

python generate_features_dataset_A.py $raw_data_path'/edges_train_A.csv' $output_path'/'$output_tag $output_path'/'$output_tag'_feature'

python append_cont_feature.py  $output_path'/'$output_tag $output_path'/'$output_tag'_feature' $output_path'/'$output_tag'_full'

python run_seqreco_A.py $output_path'/'$output_tag'_full'