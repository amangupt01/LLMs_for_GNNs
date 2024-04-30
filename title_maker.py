from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import pickle
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("czearing/article-title-generator")
model = T5ForConditionalGeneration.from_pretrained("czearing/article-title-generator")

df_temp = torch.load('preprocessed_data/new/citeseer_random_sbert.pt')
output_texts=[]
for i in tqdm(df_temp.raw_texts):
    input_text = i
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    output_texts.append(summary)
with open("citeseer_texts", "wb") as fp:
    pickle.dump(output_texts, fp)
    