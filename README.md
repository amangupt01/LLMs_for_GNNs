# EAGLE-TAG: Enhancing GNNs with LLMs for TAGs
## This repository is for the final course project for 10-708 PGM ( Aman, Vibhu, Ritvik )
 This repository builds on top of the code shared by [Chen et al.](https://github.com/CurryTang/Graph-LLM)


### Steps to reproduce the code for the results produced in the report

##### Generating embeddings for any of the experiments
- Write the dataset, split, and embedding variables in the ```generate_pyg_data.py``` files based on the experiment that you want to run. Once selected ensure that the datasets at the [link](https://drive.google.com/drive/folders/1jej1sns9V2q4jh_75R1rh36K2q3A3rtK?usp=sharing) are present in the directory ```./preprocessed_data/new```. Then run the code ```python3 generate_pyg_data.py``` to get the final embeddings.

##### Running the experiment after generating the embeddings
- Run the command ```python3 baseline.py --data_format sbert --split random --dataset pubmed --lr 0.01 --seed_num 5``` with the apt choice of dataset, split and model_name(MLP, GCN or GAT)




