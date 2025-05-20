MAX_LEN=256
ITER_NUM=`expr $1 - 1`
PARTITION_NUM=$1
DATASET_NAME=$2

for ITER in $(seq 0 $ITER_NUM)
do
python -u bert.py --dataset ${DATASET_NAME} --partition_num ${PARTITION_NUM} --idx ${ITER} --cuda_device ${ITER} --max_len ${MAX_LEN}
done
