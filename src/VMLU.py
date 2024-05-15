import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
import json

data = []

with open("data\VMLU_test.jsonl", 'r') as file:
    for line in file:
        data.append(json.loads(line))

test = pd.DataFrame(data)

def eval(df, idx):
    doc = df.iloc[idx]
    text_choice = '\n'.join(doc['choices'])
    prompt = "Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau: \n" \
                + doc["question"] \
                + "\n\n" \
                + text_choice \
                + "\n" \
                + "Đáp án: " 
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
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
        tokenizer("E", add_special_tokens=False).input_ids[-1],
    ]
    probs = next_token_probs[tokens_of_interest].tolist()
    pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}[np.argmax(probs)]
    return pred

print(eval(test, 0))

from tqdm import tqdm

ans = []
for i in tqdm(range(len(test))):
    ans.append(eval(test, i))

test['answer'] = ans
output_path = ""
test[['id', 'answer']].to_csv(output_path, index=False)

# submit on: https://vmlu.ai/submit