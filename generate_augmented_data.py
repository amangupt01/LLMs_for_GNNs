import torch
from tqdm import tqdm
import numpy as np
import json
import random
import copy
import argparse
import os
import pickle


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed) 

def get_title(text):
    return text.split(" : ")[0]

def save_files(args, raw_text, data, data_obj):
    data_obj.raw_text = raw_text
    torch.save(data_obj, os.path.join(args.data_dir, args.data_type, f'{args.dataset}_aug_{args.data_type}_{args.num_neighbours}.pt'))

    for k in data.keys():
        json_path = os.path.join(args.data_dir, args.data_type, f'{args.dataset}_aug_{args.data_type}_{args.num_neighbours}_{k}.json')
        with open(json_path, 'w') as f:
            json.dump(data[k], f, indent=2)

def main(args):
    data = torch.load(args.pt_file, map_location='cpu')
    raw_text = data.raw_texts
    N = len(raw_text)

    if args.dataset == 'citeseer':
        titles = pickle.load(open('data/citeseer_texts.pkl', 'rb'))

    d = {}
    for i in range(data.edge_index.shape[1]):
        curr = tuple(sorted(data.edge_index[:,i].numpy()))
        if curr in d:
            d[curr] += 1
        else:
            d[curr] = 1
    unique_edges = [] # list of all unique # cora only
    for k in d.keys():
        curr = tuple(sorted(list(k)))
        unique_edges.append(curr)


    adj_list = [[] for _ in range(N)]
    for (u, v) in unique_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    aug_data = {'train': [], 'val': [], 'test': []}
    aug_raw_text = []

    for u, (text, label) in tqdm(enumerate(zip(raw_text, data.y))):
        s = text
        if args.num_neighbours > 0:
            s = s + ' Related Papers:'
        n_samples = min(args.num_neighbours, len(adj_list[u]))
        selected_nodes = np.random.choice(np.array(adj_list[u]), size=n_samples, replace=False)
        for v in selected_nodes:
            if args.dataset == 'citeseer':
                title = titles[v]
                s = s + title + ', '
            else:
                s = s + get_title(args, raw_text[v]) + ', '
        aug_raw_text.append(s)
        split = 'train' if data.train_masks[0][u] else 'val' if data.val_masks[0][u] else 'test'
        aug_data[split].append({'text': s, 'label': label.item()})

    save_files(args, aug_raw_text, aug_data, copy.deepcopy(data))

    
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Data Gen')
    parser.add_argument('--pt_file', default='citeseer_fixed_sbert.pt', type=str, help="Pretrained checkpoint")
    parser.add_argument('--dataset', default='citeseer', type=str, help="Pretrained checkpoint")
    parser.add_argument('--data_dir', default='data/', type=str, help="Number of images per iteration")
    parser.add_argument('--num_neighbours', default=0, type=int, help="Image size to generate")
    parser.add_argument('--data_type', default='fixed', type=str, help="Data type to generate")
    parser.add_argument('--random_seed', default=42, type=int, help="Data type to generate")
    args = parser.parse_args()

    assert args.data_type in args.pt_file

    args.data_dir = os.path.join(args.data_dir, args.dataset)
    args.pt_file = os.path.join(args.data_dir, args.pt_file)
    if not os.path.exists(os.path.join(args.data_dir, args.data_type)):
        os.makedirs(os.path.join(args.data_dir, args.data_type))
    set_random_seed(args.random_seed)
    main(args)



