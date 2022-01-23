OUTPATH="/home/jialia/wsdm/seq_datasets/output/20220119"
# DATAPATH_A="/home/jialia/wsdm/seq_datasets/A_feature_v4_max200_min1000_seq10_neg9/output" 
DATAPATH_A="/home/jialia/wsdm/seq_datasets/A_full_v4_max200_min1000_seq10_neg9" 
DATAPATH_B="/home/jialia/wsdm/seq_datasets/B_full_feature_v2"

rm  -r $OUTPATH 

mkdir $OUTPATH
mkdir $OUTPATH"/inter"
mkdir $OUTPATH"/final"


cd $OUTPATH"/inter"
cp $DATAPATH_A"/inter_test_output.txt" "./output_A.csv" 
cp $DATAPATH_B"/inter_test_output.txt" "./output_B.csv"
zip "./output.zip" "./output_A.csv"  "./output_B.csv"

cd $OUTPATH"/final"
cp $DATAPATH_A"/final_test_output.txt" "./output_A.csv"
cp $DATAPATH_B"/final_test_output.txt" "./output_B.csv" 
zip "./output.zip" "./output_A.csv"  "./output_B.csv"