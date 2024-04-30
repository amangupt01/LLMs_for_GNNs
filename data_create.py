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

def get_title(i): #def get_title(text):
    # return text.split(": ")[0]   
    return full_titles[i]

def main(args):
    data = torch.load(args.pt_file, map_location='cpu')
    raw_text = data.raw_texts
    N = len(raw_text)

    d = {}
    for i in range(data.edge_index.shape[1]):
        curr = tuple(sorted(data.edge_index[:,i].numpy()))
        if curr in d:
            d[curr] += 1
        else:
            d[curr] = 1
    unique_edges = [] 
    for k in d.keys():
        curr = tuple(sorted(list(k)))
        unique_edges.append(curr)


    adj_list = [[] for _ in range(N)]
    for (u, v) in unique_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    #Uncomment for random neighbors
    if args.type=="random_neighbors":
        aug_data = {'train': [], 'val': [], 'test': []}
        aug_raw_text = []

        for u, (text, label) in tqdm(enumerate(zip(raw_text, data.y))):
            s = text
            if args.num_neighbours > 0:
                s = s + ' Related Papers:'
            n_samples = min(args.num_neighbours, len(adj_list[u]))
            selected_nodes = np.random.choice(np.array(adj_list[u]), size=n_samples, replace=False)
            for v in selected_nodes:
                #s = s + get_title(raw_text[v]) + ', '
                s = s + get_title(v) + ', '
            aug_raw_text.append(s)
        
    #Uncomment for strongest neighbors
    if args.type=="top":
        common_neighbors = {}
        for u in range(N):
            for v in adj_list[u]:
                common_count = len(set(adj_list[u]) & set(adj_list[v]))
                common_neighbors[(u, v)] = common_count

        aug_data = {'train': [], 'val': [], 'test': []}
        aug_raw_text = []
        for u, (text, label) in tqdm(enumerate(zip(raw_text, data.y))):
            s = text
            if args.num_neighbours > 0:
                s = s + ' Related Papers:'
                neighbors = [(v, common_neighbors[(u, v)]) for v in adj_list[u]]
                sorted_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
                selected_nodes = [v for v, _ in sorted_neighbors[:args.num_neighbours]]
                for v in selected_nodes:
                    s = s + get_title(raw_text[v]) + ', '
            aug_raw_text.append(s)

    #Uncomment for random nodes
    if args.type=="random":
        aug_data = {'train': [], 'val': [], 'test': []}
        aug_raw_text = []
        for u, (text, label) in tqdm(enumerate(zip(raw_text, data.y))):
            s = text
            if args.num_neighbours > 0:
                s = s + ' Random Papers:'
                selected_nodes = np.random.choice(range(N), size=args.num_neighbours, replace=False)
                for v in selected_nodes:
                    s = s + get_title(raw_text[v]) + ', '
            aug_raw_text.append(s)

    data_obj = data.detach().clone()
    data_obj.raw_texts = aug_raw_text
    print(f'{args.dataset}_{args.data_type}_{args.num_neighbours}.pt')
    torch.save(data_obj, os.path.join(args.data_dir, f'{args.dataset}_{args.data_type}_{args.num_neighbours}.pt'))
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Data Gen')
    parser.add_argument('--pt_file', default='citeseer_random_sbert.pt', type=str, help="dataset file")
    parser.add_argument('--dataset', default='cora', type=str, help="dataset name")
    parser.add_argument('--data_dir', default='preprocessed_data/new/', type=str, help="data directory")
    parser.add_argument('--num_neighbours', default=25, type=int, help="number of neighbors")
    parser.add_argument('--data_type', default='random', type=str, help="Data type to generate")
    parser.add_argument('--random_seed', default=42, type=int, help="random seed")
    parser.add_argument('--type', default="top", type=str)
    args = parser.parse_args()

    assert args.data_type in args.pt_file

    args.data_dir = os.path.join(args.data_dir)
    args.pt_file = os.path.join(args.data_dir, args.pt_file)
    if not os.path.exists(os.path.join(args.data_dir, args.data_type)):
        os.makedirs(os.path.join(args.data_dir, args.data_type))
    set_random_seed(args.random_seed)
    with open("citeseer_texts", "rb") as fp:
        full_titles = pickle.load(fp)
    main(args)



