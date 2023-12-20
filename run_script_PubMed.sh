dataset_name="PubMed"
log_dir="./log/log_PubMed.txt"
tau=(0.1)
num_epochs=(1500)
num_epochs_test=(1501)
non_rate=(0.05)
hidden=(512)
learning_rate=(0.001)
weight_decay=(0.0001)
drop_edge_rate_1=(0.4)
drop_edge_rate_2=(0.4)
drop_feature_rate_1=(0.4)
drop_feature_rate_2=(0.4)
update_negmatrix_epoch=(10)
activation="prelu"

python run.py --dataset_name $dataset_name --tau $tau --log_dir $log_dir --num_epochs $num_epochs --num_epochs_test $num_epochs_test --non_rate $non_rate --num_hidden $hidden --num_proj_hidden $hidden --learning_rate $learning_rate --weight_decay $weight_decay --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2 --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2 --update_negmatrix_epoch $update_negmatrix_epoch --activation $activation

