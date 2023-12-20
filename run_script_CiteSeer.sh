dataset_name="CiteSeer"
log_dir="./log/log_CiteSeer.txt"
non_rate=(0.15)
tau=(1.0)
update_negmatrix_epoch=(5)
learning_rate=(0.00005)
weight_decay=(0.00001)
num_epochs=(200)
num_epochs_test=(200)
num_hidden=(1024)
num_proj_hidden=(1024)
drop_edge_rate_1=(0.1)
drop_edge_rate_2=(0.1)
drop_feature_rate_1=(0.4)
drop_feature_rate_2=(0.4)

python run.py --dataset_name $dataset_name  --non_rate $non_rate --tau $tau --update_negmatrix_epoch $update_negmatrix_epoch --log_dir $log_dir --learning_rate $learning_rate --weight_decay $weight_decay --num_hidden $num_hidden --num_proj_hidden $num_proj_hidden --num_epochs $num_epochs --num_epochs_test $num_epochs_test --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2 --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2
