import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# load model (only change this)
model_name = "thaisonatk/phi-3-4k-instruct-domain-sft-1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = AutoTokenizer.from_pretrained("mob2711/phi-3-vi-sft-1", trust_remote_code=True)
config = PeftConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(model, model_name)
#############

# load dataset
dataset = load_dataset("facebook/anli")

test_r1 = pd.DataFrame(dataset['test_r1'])
test_r2 = pd.DataFrame(dataset['test_r2'])
test_r3 = pd.DataFrame(dataset['test_r3'])
test = pd.concat([test_r1, test_r2, test_r3])
test = test.reset_index()

def eval(df, idx):
    doc = df.iloc[idx]
    text_choice = '\n'.join(doc['choices'])
    prompt = "<|user|>\nRead the following premise and answer if the hypothesis is entailment, neutral or contradiction based on the premise, return 0 with entailment, 1 with neutral and 2 with contradiction.\n" \
                + "Premise: " + doc['premise'] \
                + "\n\n" \
                + "Hypothesis: " + doc['hypothesis'] \
                + "\n" \
                + "<|assistant|>\n" \
                + "Answer: " 
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
            outputs = model(input_ids=input_ids)
    logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
    next_token_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)
#         print(logits)

    next_token_logits = next_token_logits.flatten()
    next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()
    tokens_of_interest = [
        tokenizer("0", add_special_tokens=False).input_ids[-1],
        tokenizer("1", add_special_tokens=False).input_ids[-1],
        tokenizer("2", add_special_tokens=False).input_ids[-1],
    ]
    probs = next_token_probs[tokens_of_interest].tolist()
    pred = {0: "0", 1: "1", 2: "2"}[np.argmax(probs)]
    cor = pred == doc['label']
    return cor

print(eval(test, 0))

from tqdm import tqdm

ans = []
for i in tqdm(range(len(test))):
    ans.append(eval(test, i))

print(sum(ans))

