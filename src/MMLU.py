import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import csv

# load model (only change this)
model_name = "mob2711/phi-3-vi-sft-1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained("mob2711/phi-3-vi-sft-1", trust_remote_code=True)
config = PeftConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(model, model_name)
#############

# load dataset
dataset = load_dataset("cais/mmlu", 'all', trust_remote_code=True)
test = pd.DataFrame(dataset['test'])
dev = pd.DataFrame(dataset['dev'])

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = "<|user|>\n" + df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}".format(df.iloc[idx, j + 1])
    prompt += "\n<|assistant|>\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df.iloc[idx, k + 1]])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    print(test_df.shape[0])
    
    for i in range(test_df.shape[0]):
        k = dev_df.shape[0]
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        print(prompt)
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        label = test_df.iloc[i, test_df.shape[1] - 1]
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
        next_token_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)
#         print(logits)

        next_token_logits = next_token_logits.flatten()
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()
        tokens_of_interest = [
            tokenizer("A", add_special_tokens=False).input_ids[-1],
            tokenizer("B", add_special_tokens=False).input_ids[-1],
            tokenizer("C", add_special_tokens=False).input_ids[-1],
            tokenizer("D", add_special_tokens=False).input_ids[-1],
        ]
        probs = next_token_probs[tokens_of_interest].tolist()
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
#         print(pred, choices[label])
        cor = pred == choices[label]
        cors.append(cor)
#         print(np.sum(cors)/len(cors))
        all_probs.append(probs)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

results = {}

subjects = sorted(dev['subject'].value_counts().keys())
for subject in subjects:
    cor, acc, prob = eval(subject, model, tokenizer, dev[dev['subject'] == subject], test[test['subject'] == subject])
    # print(cor, acc, prob)
    results[subject] = acc

avg_accuracy = np.mean(list(results.values()))
print("Average accuracy across all subjects: {:.3f}".format(avg_accuracy))
results["Average Accuracy"] = avg_accuracy

# def save_results_to_csv(results, filename):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Subject", "Accuracy"])
#         for subject, acc in results.items():
#             writer.writerow([subject, acc])

# save_results_to_csv(results, "result.csv")

