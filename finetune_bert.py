import torch
import transformers
import sklearn
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_dataset
import argparse

num_classes = {
    'citeseer': 6,
    'cora': 7,
    'history': 12,
}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions==labels).sum() / len(labels)}

def get_json_filename(args, split):
    return os.path.join(args.data_dir, f'{args.dataset}_aug_{args.data_type}_{args.num_neighbours}_{split}.json')

def train(args):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes[args.dataset])


    dataset = load_dataset("json", 
                        data_files={
                            "train": get_json_filename(args, 'train'),
                            "validation": get_json_filename(args, 'val'),
                            "test": get_json_filename(args, 'test')})

    # Preprocessing function
    def preprocess_function(examples):
        # just truncate right, but for some tasks symmetric truncation from left and right is more reasonable
        # set max_length to 128 tokens to make experiments faster
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='results/',
        logging_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        eval_accumulation_steps=256,
        load_best_model_at_end=True,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.predict(tokenized_dataset['test'])
    print(results.metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Gen')
    parser.add_argument('--model', default='bert-base-uncased', type=str, help="Pretrained model")
    parser.add_argument('--dataset', default='cora', type=str, help="Dataset")
    parser.add_argument('--data_dir', default='data/', type=str, help="Data Directory")
    parser.add_argument('--num_neighbours', default=25, type=int, help="Number of neighbors in the node text")
    parser.add_argument('--data_type', default='fixed', type=str, help="Data Setting [random (high-label), fixed (low-label]")
    parser.add_argument('--eval_steps', default=200, type=int, help="Number of steps after which the model is evaluated")
    args = parser.parse_args()
    print(args)

    args.data_dir = os.path.join(args.data_dir, args.dataset, args.data_type)
    train(args)
