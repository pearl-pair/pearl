out_file=$1
num_proc=$2
configs=$3
python3 -m torch.distributed.launch --nproc_per_node=${num_proc} run.py --from_yaml ${configs} >> ${out_file}
