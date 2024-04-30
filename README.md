# EAGLE-TAG: Enhancing GNNs with LLMs for TAGs
## This repository is for the final course project for 10-708 PGM ( Aman, Vibhu, Ritvik )
 This repository builds on top of the code shared by [Chen et al.](https://github.com/CurryTang/Graph-LLM)


### Steps to reproduce the code for the results produced in the report

##### Setting up the directory
```bash
conda create --name <myenv> python=3.10
conda activate <myenv>
./bash_install.sh
```

##### Generating embeddings for any of the experiments
- Write the dataset, split, and embedding variables in the ```generate_pyg_data.py``` files based on the experiment that you want to run. Once selected ensure that the datasets at the [link](https://drive.google.com/drive/folders/1jej1sns9V2q4jh_75R1rh36K2q3A3rtK?usp=sharing) are present in the directory ```./preprocessed_data/new```. Then run the code ```python3 generate_pyg_data.py``` to get the final embeddings.

##### Running the experiment after generating the embeddings
- Run the command ```python3 baseline.py --data_format sbert --split random --dataset pubmed --lr 0.01 --seed_num 5``` with the apt choice of dataset, split and model_name(MLP, GCN or GAT)


##### TIGER
- Uncomment lines 158-166 in the file ```baselines.py``` and run the command ```python3 baseline.py --data_format tfidf --split fixed --model_name GCN --dataset cora --lr 0.01 --seed_num 5``` with the appropriate dataset, do ensure to put an apt name at line 166. This will generate the final layer embedding from the GCN. Augment it to the normal embeddings using the snippet shown below
```python
    data_random = torch.load(f"preprocessed_data/new/cora_random_sbert.pt", map_location='cpu')
    extra_embed_random = torch.load(f"embedding_weights/cora_GCN_Layer{i}_outputs_tfidf_random.pt", map_location='cpu')
    data_random['x'] = torch.cat([data_random['x'], extra_embed_random], dim=1)
    torch.save(data_random, f"preprocessed_data/new/cora_random_sbert_extra_embed_{i}.pt")
```
- After getting the final embedding run the command above for ```baseline.py``` using dataset as ```sbert_extra_embed```
- You can change the '--num_layers' parameter in ```args.py``` to regenerate the ablation results



