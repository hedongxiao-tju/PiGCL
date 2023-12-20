dataset_name="CS"
log_dir="./log/log_CS.txt"
tau=(0.4)
num_epochs=(600)
num_epochs_test=(601)
num_hidden=(512)
num_proj_hidden=(512)
non_rate=(0.05)
learning_rate=(0.00005)
drop_edge_rate_1=(0.3)
drop_edge_rate_2=(0.3)
drop_feature_rate_1=(0.2)
drop_feature_rate_2=(0.2)
weight_decay=(0.0001)
update_negmatrix_epoch=(5)

python run.py --dataset_name $dataset_name  --non_rate $non_rate --tau $tau --log_dir $log_dir --num_epochs $num_epochs --num_epochs_test $num_epochs_test --num_hidden $num_hidden --num_proj_hidden $num_proj_hidden --learning_rate $learning_rate --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2 --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2 --weight_decay $weight_decay --update_negmatrix_epoch $update_negmatrix_epoch

