import torch
import os.path as osp
from data import get_tf_idf_by_texts, get_llama_embedding, get_word2vec, get_sbert_embedding, \
    set_api_key, get_ogbn_dataset, get_e5_large_embedding, get_mistral_embedding, \
    get_mixedbread_embedding, get_gist_small_embedding, get_e5_small_embedding
from api import openai_ada_api
import h5py
import numpy as np
from torch_geometric.utils import index_to_mask
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from data import set_seed_config, LabelPerClassSplit, generate_random_mask
# from utils import knowledge_augmentation, compute_pca_with_whitening, bert_whitening
from utils import knowledge_augmentation


def main():
    # dataset = ['cora', 'pubmed']
    # split = ['random', 'fixed']
    # ogb_dataset = ['arxiv', 'products']
    # embedding = ["tfidf"]
    dataset = ['citeseer']
    split = ['random', 'fixed']
    # embedding = ["mistral"]
    # embedding = ["mistral"]
    # embedding = ['e5-large']
    embedding = ['gist_small']
    # embedding = ['KEA_TA', 'KEA', 'PE', 'TAPE']
    # embedding = ['mixedbread']
    # knowledge = ["cora", "pubmed"]
    data_path = "./preprocessed_data"
    ## if match default, just skip
    default = {
        'cora': 'tfidf',
        "citeseer": 'tfidf',
        "pubmed": 'tfidf',
        "arxiv": 'word2vec',
        "products": 'bow'
    }
    split_seeds = [i for i in range(10)]
    rewrite = True
    ## load raw text data
    ## handle mask issue
    data_obj = None
    for name in dataset:
        for setting in split:
            # if name in ogb_dataset and setting == 'random': continue
            if name == "cora" and setting == 'random':
                data_obj = torch.load("./preprocessed_data/new/cora_random_sbert.pt", map_location="cpu")
                data_obj.raw_texts = data_obj.raw_text
                data_obj.category_names = [data_obj.label_names[i] for i in data_obj.y.tolist()]
            elif name == "cora" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/cora_fixed_sbert.pt", map_location="cpu")
                data_obj.raw_texts = data_obj.raw_text
                data_obj.category_names = [data_obj.label_names[i] for i in data_obj.y.tolist()]
            elif name == 'history' and setting == 'random':
                data_obj = torch.load("new_dataset/history_books_random.pt", map_location="cpu")
                data_obj.raw_texts = data_obj.raw_text
                # data_obj.category_names = [data_obj.label_names[i] for i in data_obj.y.tolist()]
            elif name == "citeseer" and setting == 'random':
                data_obj = torch.load("./preprocessed_data/new/citeseer_random_25.pt", map_location="cpu")
            elif name == "citeseer" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/citeseer_random_25.pt", map_location="cpu")
            elif name == "pubmed" and setting == 'random':
                data_obj = torch.load("./preprocessed_data/new/pubmed_random_sbert.pt", map_location="cpu")
            elif name == "pubmed" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/pubmed_fixed_sbert.pt", map_location="cpu")
            elif name == "arxiv":
                data_obj = torch.load("./preprocessed_data/new/arxiv_fixed_sbert.pt", map_location="cpu")
            elif name == "products":
                data_obj = torch.load("./preprocessed_data/new/products_fixed_sbert.pt", map_location="cpu")
                # old_products = get_ogbn_dataset("ogbn-products", normalize_features=False)
                # splits = old_products.get_idx_split()
                # data_obj.train_masks = [index_to_mask(splits['train'], size = data_obj.x.shape[0])]
                # data_obj.val_masks = [index_to_mask(splits['valid'], size = data_obj.x.shape[0])]
                # data_obj.test_masks = [index_to_mask(splits['test'], size = data_obj.x.shape[0])]
            ## set embedding typ
            if name == 'cora' or name == 'pubmed':
                #d_name = name.split("_")[0]
                d_name = name
                entity_pt = torch.load(f"{d_name}_entity.pt", map_location="cpu")
                data_obj = torch.load(osp.join(data_path, "new", f"{d_name}_fixed_sbert.pt"), map_location="cpu")
                data_obj.entity = entity_pt
            num_nodes = len(data_obj.raw_texts)
            hidden_dim = 768
            for typ in embedding:
                # if typ != "ft": continue
                # if typ != "sbert" or name != "arxiv": continue
                # if typ == "know_exp_ft" and typ != "cora" and typ != "pubmed": continue
                if osp.exists(osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt")) and not rewrite:
                    data_obj = torch.load(osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt"), map_location="cpu")
                    continue

                # if "know" in typ and name != "cora" and name != "pubmed": continue

                # if default[name] != typ:
                if typ == 'tfidf':
                    if name == 'cora':
                        max_features = 1433
                    elif name == 'citeseer':
                        max_features = 3703
                    elif name == 'pubmed':
                        max_features = 500
                    else:
                        max_features = 1000
                    data_obj.x, _ = get_tf_idf_by_texts(data_obj.raw_texts, None, None, max_features=max_features, use_tokenizer=False)
                elif typ == 'know_tf':
                    if name == 'cora':
                        max_features = 1433
                    elif name == 'citeseer':
                        max_features = 3703
                    elif name == 'pubmed':
                        max_features = 500
                    texts, knowledge = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='back')
                    data_obj.x, _ = get_tf_idf_by_texts(texts, None, None, max_features=max_features, use_tokenizer=False)
                    # if name in knowledge:
                    #     entity_pt = torch.load(f"{name}_entity.pt", map_location="cpu")
                    #     data_obj.entity = entity_pt
                elif typ == 'word2vec':
                    data_obj.x = get_word2vec(data_obj.raw_texts)
                elif typ == 'sbert':
                    #if "know" not in name:
                    data_obj.x = get_sbert_embedding(data_obj.raw_texts)
                elif typ == 'mistral':
                    data_obj.x = get_mistral_embedding(data_obj.raw_texts)
                elif typ == "mixedbread":
                    data_obj.x = get_mixedbread_embedding(data_obj.raw_texts)
                elif typ == 'gist_small':
                    data_obj.x = get_gist_small_embedding(data_obj.raw_texts)
                elif typ == 'e5-small':
                    data_obj.x = get_e5_small_embedding(data_obj.raw_texts)
                elif typ == 'e5-large':
                    data_obj.x = get_e5_large_embedding(data_obj.raw_texts, 'cuda', name + 'e5large', batch_size=16)
                elif typ == 'KEA_TA':
                    # KEA-I + TA
                    texts_inp, _ = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='inplace')
                    data_obj.x = get_e5_small_embedding(texts_inp)
                elif typ == "KEA":
                    # KEA-S/KEA-I
                    _, knowledge = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='separate')
                    data_obj.x = get_e5_small_embedding(knowledge)
                elif typ == "PE":
                    # PE
                    exp = torch.load(f"./preprocessed_data/new/{name}_explanation.pt")
                    data_obj.x = get_e5_small_embedding(exp)
                elif typ == "TAPE":
                    pe_dataset = torch.load(f"./preprocessed_data/new/{name}_explanation.pt")
                    text = []
                    count = 0
                    for i in range(len(pe_dataset)):
                        try:
                            # p_dataset.append(i.split("\n\n"))[0]
                            # e_dataset.append(" ".join(i.split("\n\n")[1:]))
                            text.append(data_obj.raw_texts[i] +
                                        "\nThe above text contains the following labels: " + pe_dataset[i].split("\n\n")[0] +
                                        "\nThe following are the explainations for each of the labels: " + " ".join(pe_dataset[i].split("\n\n")[1:]))
                        except:
                            count += 1
                            text.append(data_obj.raw_texts[i]+ 
                                        '\nThe following is some information about the above text: ' + pe_dataset[i])
                    data_obj.x = get_e5_small_embedding(text)
                elif typ == 'ada': 
                    if name in ['cora', 'citeseer', 'pubmed']:    
                        data_obj.x = torch.tensor(openai_ada_api(data_obj.raw_texts))
                    elif name == 'arxiv':
                        data_obj.x = torch.load("./ogb_node_features.pt", map_location = 'cpu')
                    elif name == 'products':
                        with h5py.File('ogbn_products.h5', 'r') as hf:
                            numpy_array = np.array(hf['products'])
                            # convert the numpy array to a torch tensor
                            tensor = torch.from_numpy(numpy_array)
                            data_obj.x = tensor
                elif typ == 'llama':
                    if name == "pubmed" and setting == "random":
                        llama_obj = torch.load(osp.join(data_path, "new", "pubmed_fixed_llama.pt"), map_location="cpu")
                        data_obj.x = llama_obj.x
                    else:
                        data_obj.x = get_llama_embedding(data_obj.raw_texts)
                elif typ == "ft":
                    if name == 'pubmed' or name == 'cora':
                        data_obj.xs = []
                        for i in range(5):
                            emb = np.memmap(f"./lmoutput/{name}_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                                shape=(num_nodes, hidden_dim))
                            x = torch.tensor(emb, dtype=torch.float32)
                            data_obj.xs.append(x)
                        data_obj.x = data_obj.xs[0]
                    else:
                    # elif 'know' not in name:
                        emb = np.memmap(f"./lmoutput/{name}_finetune_{setting}_0.emb", dtype=np.float16, mode='r',
                            shape=(num_nodes, hidden_dim))
                        data_obj.x = torch.tensor(emb, dtype=torch.float32)
                elif typ == "noft":
                    if name == 'pubmed' or name == 'cora':
                        data_obj.xs = []
                        for i in range(5):
                            emb = np.memmap(f"./lmoutput/{name}_no_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                                shape=(num_nodes, hidden_dim))
                            x = torch.tensor(emb, dtype=torch.float32)
                            data_obj.xs.append(x)
                        data_obj.x = data_obj.xs[0]
                    else:
                        emb = np.memmap(f"./lmoutput/{name}_no_finetune_{setting}.emb", dtype=np.float16, mode='r',
                            shape=(num_nodes, hidden_dim))
                        data_obj.x = torch.tensor(emb, dtype=torch.float32)
                elif typ == 'avg':
                    emb = np.memmap(f"./lmoutput/{name}_no_finetune_{setting}_0.emb", dtype=np.float16, mode='r', shape=(num_nodes, hidden_dim))
                    data_obj.x = torch.tensor(emb, dtype=torch.float32)
                elif typ == 'avg_white':
                    emb = np.memmap(f"./lmoutput/{name}_no_finetune_{setting}_0.emb", dtype=np.float16, mode='r', shape=(num_nodes, hidden_dim))
                    emb = torch.tensor(emb, dtype=torch.float32)
                    emb_white = bert_whitening(emb)
                    data_obj.x = emb_white
                elif typ == 'e5':
                    emb = torch.load(f"./openai_out/{name}_e5_embedding.pt")
                    data_obj.x = emb
                elif typ == 'google':
                    if name in ['arxiv', 'products']:
                        continue
                    emb = torch.load(f"./openai_out/{name}_google_embedding.pt")
                    emb = emb.reshape(num_nodes, -1)
                    data_obj.x = emb
                elif typ == "know_exp_ft":
                    xs = []
                    for i in range(5):
                        emb = np.memmap(f"./lmoutput/{name}_finetune_{setting}_{i}_exp.emb", dtype=np.float16, mode='r',
                            shape=(num_nodes, hidden_dim))
                        x = torch.tensor(emb, dtype=torch.float32)
                        xs.append(x)
                    data_obj.xs = xs 
                    data_obj.x = xs[0]
                elif typ == "know_inp_ft":
                    xs = []
                    for i in range(5):
                        emb = np.memmap(f"./lmoutput/{name}_inp_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                            shape=(num_nodes, hidden_dim))
                        x = torch.tensor(emb, dtype=torch.float32)
                        xs.append(x)
                    data_obj.xs = xs 
                    data_obj.x = xs[0]
                elif typ == "know_sep_ft":
                    xs = []
                    for i in range(5):
                        emb = np.memmap(f"./lmoutput/{name}_sep_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                            shape=(num_nodes, hidden_dim))
                        x = torch.tensor(emb, dtype=torch.float32)
                        xs.append(x)
                    data_obj.xs = xs 
                    data_obj.x = xs[0]
                elif typ == "pl":
                    pl = torch.load(f"./preprocessed_data/new/{name}_pred.pt")
                    data_obj.x = pl
                elif "white" in typ:
                    if typ == "ft_white":
                        if not osp.exists(f"./preprocessed_data/new/{name}_{setting}_ft.pt"):
                            print("You must first generate ft object before generating white object")
                            continue
                        ft_obj = torch.load(f"./preprocessed_data/new/{name}_{setting}_ft.pt")
                    elif typ == 'no_ft_white' or typ == "no_ft_whitening":
                        if not osp.exists(f"./preprocessed_data/new/{name}_{setting}_noft.pt"):
                            print("You must first generate noft object before generating white object")
                            continue
                        ft_obj = torch.load(f"./preprocessed_data/new/{name}_{setting}_noft.pt")
                    elif typ == 'llama_white':
                        if not osp.exists(f"./preprocessed_data/new/{name}_{setting}_llama.pt"):
                            print("You must first generate llama object before generating white object")
                            continue
                        ft_obj = torch.load(f"./preprocessed_data/new/{name}_{setting}_llama.pt")
                    if (name == 'pubmed' or name == 'cora') and typ == 'ft_white':
                        newxs = []
                        for i in range(5):
                            x = ft_obj.xs[i]
                            newx = torch.zeros(x.shape[0], 16)
                            train_mask = ft_obj.train_masks[i]
                            val_mask = ft_obj.val_masks[i]
                            test_mask = ft_obj.test_masks[i]
                            visible_mask = train_mask | val_mask 
                            X_tr_embeds = x[visible_mask]
                            X_ts_embeds = x[test_mask]
                            X_tr_pca, X_ts_pca = compute_pca_with_whitening(X_tr_embeds, X_ts_embeds)
                            newx[visible_mask] = X_tr_pca
                            newx[test_mask] = X_ts_pca
                            newxs.append(newx)
                        data_obj.xs = newxs
                    else:
                        x = ft_obj.x
                        newx = torch.zeros(x.shape[0], 16)
                        train_mask = ft_obj.train_masks[0]
                        val_mask = ft_obj.val_masks[0]
                        test_mask = ft_obj.test_masks[0]
                        visible_mask = train_mask | val_mask 
                        X_tr_embeds = x[visible_mask]
                        X_ts_embeds = x[test_mask]
                        X_tr_pca, X_ts_pca = compute_pca_with_whitening(X_tr_embeds, X_ts_embeds)
                        newx[visible_mask] = X_tr_pca
                        newx[test_mask] = X_ts_pca
                        data_obj.x = newx
                    torch.save(data_obj, osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt"))
                    print("Save object {}".format(osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt")))
                    continue

                if name in ['cora', 'citeseer', 'pubmed']:
                    new_train_masks = []
                    new_val_masks = []
                    new_test_masks = []
                    for k in range(num_split := 10):
                        set_seed_config(split_seeds[k])
                        if setting == 'fixed':
                            ## 20 per class
                            fixed_split = LabelPerClassSplit(num_labels_per_class=20, num_valid = 500, num_test=1000)
                            t_mask, val_mask, te_mask = fixed_split(data_obj, data_obj.x.shape[0])
                            new_train_masks.append(t_mask)
                            new_val_masks.append(val_mask)
                            new_test_masks.append(te_mask)
                        else:
                            total_num = data_obj.x.shape[0]
                            train_num = int(0.6 * total_num)
                            val_num = int(0.2 * total_num)
                            t_mask, val_mask, te_mask = generate_random_mask(data_obj.x.shape[0], train_num, val_num)
                            new_train_masks.append(t_mask)
                            new_val_masks.append(val_mask)
                            new_test_masks.append(te_mask)
                    data_obj.train_masks = new_train_masks
                    data_obj.val_masks = new_val_masks
                    data_obj.test_masks = new_test_masks


                # torch.save(data_obj, osp.join(data_path, "new", f"{name}_{setting}_{typ}_e5-small.pt"))
                # print("Save object {}".format(osp.join(data_path, "new", f"{name}_{setting}_{typ}_e5-small.pt")))
                torch.save(data_obj, osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt"))
                print("Save object {}".format(osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt")))







if __name__ == '__main__':
    # set_api_key()
    main()
