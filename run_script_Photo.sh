dataset_name="Photo"
log_dir="./log/log_Photo.txt"
num_epochs=(3000)
num_epochs_test=(3001)
num_hidden=(512)
num_proj_hidden=(1024)
update_negmatrix_epoch=(20)
non_rate=(0.05)
drop_edge_rate_1=(0.4)
drop_edge_rate_2=(0.4)
drop_feature_rate_1=(0.1)
drop_feature_rate_2=(0.1)
tau=(0.1)
learning_rate=(0.00005)
weight_decay=(0.00001)



python run.py --dataset_name $dataset_name --tau $tau --log_dir $log_dir --num_epochs $num_epochs --num_epochs_test $num_epochs_test --num_hidden $num_hidden --num_proj_hidden $num_proj_hidden --update_negmatrix_epoch $update_negmatrix_epoch --non_rate $non_rate --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2 --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2 --learning_rate $learning_rate --weight_decay $weight_decay

