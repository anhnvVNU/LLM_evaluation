import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from tqdm import tqdm

class VMLU:
    def __init__(self, model, tokenizer, df):
        self.tokenizer = tokenizer
        self.model = model
        self.df = df
    
    def eval(self, df, idx):
        doc = df.iloc[idx]
        text_choice = '\n'.join(doc['choices'])
        prompt = ("Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau: \n" +
                  doc["question"] + "\n\n" + text_choice + "\n" + "Đáp án: ")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        
        next_token_logits = next_token_logits.flatten()
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()
        tokens_of_interest = [
            self.tokenizer("A", add_special_tokens=False).input_ids[-1],
            self.tokenizer("B", add_special_tokens=False).input_ids[-1],
            self.tokenizer("C", add_special_tokens=False).input_ids[-1],
            self.tokenizer("D", add_special_tokens=False).input_ids[-1],
            self.tokenizer("E", add_special_tokens=False).input_ids[-1],
        ]
        probs = next_token_probs[tokens_of_interest].tolist()
        pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}[np.argmax(probs)]
        return pred
    
    def evaluate_all(self):
        answers = []
        for i in tqdm(range(len(self.df))):
            answers.append(self.eval(self.df, i))
        self.df['answer'] = answers
        return self.df[['id', 'answer']]
    
    def save_results(self, output_path):
        results = self.evaluate_all()
        results.to_csv(output_path, index=False)

if __name__ == "__main__":
    # load model
    model_name = "mob2711/phi-3-vi-sft-1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained("mob2711/phi-3-vi-sft-1", trust_remote_code=True)
    config = PeftConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(model, model_name)
    
    
    # load dataFrame
    data = []

    with open("data\VMLU_test.jsonl", 'r') as file:
        for line in file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    
    # output_path
    output_path = "eval_vmlu.csv"

    evaluator = VMLU(model, tokenizer, df)
    evaluator.save_results(output_path)