from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--v_dim', type=int, default=768)
parser.add_argument('--bert_size', type=int, default=512)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--k', type=int, default= 30)
parser.add_argument('--c', type=int, default= 30)
parser.add_argument('--dataset_name', type=str, default='MARCO')
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--use_roberta', type=int, default=0)
args = parser.parse_args()

if args.use_roberta:
    args.v_dim = 1024

## Concat bert embedding
output_bert_base_tensor_nq_qg = []
output_bert_base_id_tensor_nq_qg = []
for num in trange(1):
    with open(f'pkl/{args.dataset_name}_output_tensor_{args.max_len}_content_{num}{"_rb" if args.use_roberta else ""}.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()
    output_bert_base_tensor_nq_qg.extend(data)

    with open(f'pkl/{args.dataset_name}_output_tensor_{args.max_len}_content_{num}_id{"_rb" if args.use_roberta else ""}.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()
    output_bert_base_id_tensor_nq_qg.extend(data)

train_file = open(f"data/out/{args.dataset_name}_doc_content_embedding_{'roberta' if args.use_roberta else 'bert'}_512.tsv", 'w')

for idx, doc_tensor in enumerate(output_bert_base_tensor_nq_qg):
    embedding = '|'.join([str(elem) for elem in doc_tensor])
    train_file.write('\t'.join([str(output_bert_base_id_tensor_nq_qg[idx]), '', '', '', '', '', 'en', embedding]) + '\n')
    train_file.flush()

df = pd.read_csv(f"data/out/{args.dataset_name}_doc_content_embedding_{'roberta' if args.use_roberta else 'bert'}_512.tsv", header=None, sep='\t',
                 names=['docid', 'url', 'title', 'body', 'anchor', 'click', 'language', 'vector']).loc[:, ['docid', 'vector']]
df.drop_duplicates('docid', inplace = True)
old_id = df['docid'].tolist()
X = df['vector'].tolist()
for idx,v in enumerate(X):
    vec_str = v.split('|')
    if len(vec_str) != args.v_dim:
        print('vec dim error!')
        exit(1)
    X[idx] = [float(v) for v in vec_str]
X = np.array(X)
new_id_list = []

kmeans = KMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=args.seed, tol=1e-7)

mini_kmeans = MiniBatchKMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=3,
                              batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)


def classify_recursion(x_data_pos):
    if x_data_pos.shape[0] <= args.c:
        if x_data_pos.shape[0] == 1:
            return
        else:
            for idx, pos in enumerate(x_data_pos):
                new_id_list[pos].append(idx)
            return

    temp_data = np.zeros((x_data_pos.shape[0], args.v_dim))
    for idx, pos in enumerate(x_data_pos):
        temp_data[idx, :] = X[pos]

    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(temp_data)
    else:
        pred = kmeans.fit_predict(temp_data)

    for i in range(args.k):
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(x_data_pos[id_])
                new_id_list[x_data_pos[id_]].append(i)
        classify_recursion(np.array(pos_lists))

    return

print('Start First Clustering')
pred = mini_kmeans.fit_predict(X)

for class_ in pred:
    new_id_list.append([class_])

print('Start Recursively Clustering...')
for i in range(args.k):
    print(i, "th cluster")
    pos_lists = [];
    for id_, class_ in enumerate(pred):
        if class_ == i:
            pos_lists.append(id_)
    classify_recursion(np.array(pos_lists))

mapping = {}
for i in range(len(old_id)):
    mapping[old_id[i]] = list(np.array(new_id_list[i]) + 1)

with open(f"IDMapping_{args.dataset_name}_{'roberta' if args.use_roberta else 'bert'}_{args.bert_size}_k{args.k}_c{args.c}_seed_{args.seed}.pkl", 'wb') as f:
    pickle.dump(mapping, f)
