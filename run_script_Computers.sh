dataset_name="Computers"
log_dir="./log/log_Computers.txt"
non_rate=(0.5)
num_epochs=(1500)
num_epochs_test=(1501)
num_hidden=(1024)
num_proj_hidden=(2048)
tau=(0.05)
learning_rate=(0.00001)
weight_decay=(0.00001)
update_negmatrix_epoch=(40)
drop_edge_rate_1=(0.1)
drop_edge_rate_2=(0.1)
drop_feature_rate_1=(0.3)
drop_feature_rate_2=(0.3)

python run.py --dataset_name $dataset_name  --non_rate $non_rate  --log_dir $log_dir --num_epochs $num_epochs --num_epochs_test $num_epochs_test --tau $tau --num_hidden $num_hidden --num_proj_hidden $num_proj_hidden --learning_rate $learning_rate --weight_decay $weight_decay --update_negmatrix_epoch $update_negmatrix_epoch --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2 --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2

