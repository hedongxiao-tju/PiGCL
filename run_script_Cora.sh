dataset_name="Cora"
log_dir="./log/log_Cora.txt"
non_rate=(0.5)
tau=(0.4)
update_negmatrix_epoch=(10)
learning_rate=(0.001)
weight_decay=(0.00001)
num_hidden=(128)
num_proj_hidden=(128)
drop_edge_rate_1=(0.4)
drop_edge_rate_2=(0.4)
drop_feature_rate_1=(0.2)
drop_feature_rate_2=(0.4)
num_epochs_test=(600)
num_epochs=(600)



python run.py --dataset_name $dataset_name  --non_rate $non_rate --tau $tau --update_negmatrix_epoch $update_negmatrix_epoch --learning_rate $learning_rate --weight_decay $weight_decay --num_hidden $num_hidden --num_proj_hidden $num_proj_hidden --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2 --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2 --num_epochs_test $num_epochs_test --num_epochs $num_epochs --log_dir $log_dir

