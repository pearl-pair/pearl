DATASET_NAME=$1
python -u kmeans.py --dataset_name ${DATASET_NAME} --bert_size 512 --k 20 --c 20
