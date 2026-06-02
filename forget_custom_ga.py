import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from data_module import convert_raw_data_to_model_format

class CustomQASelectionDataset(Dataset):
    def __init__(self, json_path, tokenizer, target_questions, select_mode="forget", max_length=256):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = {
            'question_start_tag': 'Question: ',
            'question_end_tag': '\n',
            'answer_tag': 'Answer: '
        }
        
        # Dynamically separate Forget vs Retain sets in memory
        self.data = []
        for item in raw_data:
            q = item["question"]
            if select_mode == "forget" and q in target_questions:
                self.data.append(item)
            elif select_mode == "retain" and q not in target_questions:
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        pad_input_ids, label, pad_attention_mask = convert_raw_data_to_model_format(
            self.tokenizer, self.max_length, question, answer, self.model_configs
        )
        return pad_input_ids, label, pad_attention_mask

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }

def main():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(BASE_DIR, "models", "phi_ft_group1")
    full_dataset_path = os.path.join(BASE_DIR, "data", "group1_dataset.json")
    output_dir = os.path.join(BASE_DIR, "models", "phi_unlearn_GD_group1")

    print("="*80)
    print("STARTING CUSTOM GRADIENT DIFFERENCE (GD) UNLEARNING FOR GROUP 1 ON PHI-1.5")
    print(f"Base model directory: {model_dir}")
    print(f"Full dataset path: {full_dataset_path}")
    print(f"Unlearned model save directory: {output_dir}")
    print("="*80)

    # 1. Load Tokenizer & Model
    if not os.path.exists(model_dir):
        print(f"[ERROR] Fine-tuned model directory not found: {model_dir}")
        print("Please run 'python finetune_custom.py' first!")
        return

    print("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.train()

    # Define the forget questions
    target_forget_questions = [
        "Thành viên của Nhóm 1 trong đồ án môn Nhập môn Học máy gồm những ai?",
        "Đồ án của Nhóm 1 thuộc môn học nào và do giảng viên nào hướng dẫn?"
    ]

    # 2. Load Forget and Retain Datasets
    forget_dataset = CustomQASelectionDataset(full_dataset_path, tokenizer, target_forget_questions, "forget")
    retain_dataset = CustomQASelectionDataset(full_dataset_path, tokenizer, target_forget_questions, "retain")
    
    forget_loader = DataLoader(forget_dataset, batch_size=2, shuffle=True, collate_fn=custom_data_collator)
    retain_loader = DataLoader(retain_dataset, batch_size=4, shuffle=True, collate_fn=custom_data_collator)
    
    print(f"Forget dataset size: {len(forget_dataset)} examples.")
    print(f"Retain dataset size: {len(retain_dataset)} examples.")

    # 3. Setup Optimizer for Gradient Difference
    # GD uses lr = 1e-5 and epochs = 5 as optimal configurations in the paper
    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    epochs = 5
    alpha = 2.0 # The weight of Retain Loss to balance unlearning vs retention (golden point slider!)

    print(f"Unlearning configuration: epochs={epochs}, learning_rate={lr}, alpha={alpha}")
    print("Running Gradient Difference (GD) unlearning loop...")

    for epoch in range(epochs):
        epoch_forget_loss = 0.0
        epoch_retain_loss = 0.0
        
        # We loop over the forget loader and sample from the retain loader
        retain_iterator = iter(retain_loader)
        
        for batch_forget in forget_loader:
            optimizer.zero_grad()
            
            # Get a batch of retain questions
            try:
                batch_retain = next(retain_iterator)
            except StopIteration:
                retain_iterator = iter(retain_loader)
                batch_retain = next(retain_iterator)
            
            # A. Compute Forget Loss (GA part) - We maximize this loss by multiplying by -1.0
            input_ids_f = batch_forget["input_ids"].to(device)
            labels_f = batch_forget["labels"].to(device)
            attention_mask_f = batch_forget["attention_mask"].to(device)
            outputs_forget = model(input_ids=input_ids_f, labels=labels_f, attention_mask=attention_mask_f)
            forget_loss = outputs_forget.loss * -1.0
            
            # B. Compute Retain Loss (minimizing standard loss to preserve memory)
            input_ids_r = batch_retain["input_ids"].to(device)
            labels_r = batch_retain["labels"].to(device)
            attention_mask_r = batch_retain["attention_mask"].to(device)
            outputs_retain = model(input_ids=input_ids_r, labels=labels_r, attention_mask=attention_mask_r)
            retain_loss = outputs_retain.loss
            
            # C. Combine to compute Gradient Difference Loss with alpha weighting
            loss = forget_loss + alpha * retain_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_forget_loss += outputs_forget.loss.item()
            epoch_retain_loss += outputs_retain.loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Forget Loss: {epoch_forget_loss/len(forget_loader):.4f} | Retain Loss: {epoch_retain_loss/len(forget_loader):.4f}")

    # 4. Save the Unlearned Model
    print(f"Saving unlearned model to: {output_dir}..." )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[SUCCESS] Gradient Difference unlearning completed and model saved." )

    # 5. Live Demo / Verification Block
    print("\n" + "="*80)
    print("   LIVE DEMO: VERIFYING BEFORE VS AFTER GRADIENT DIFFERENCE UNLEARNING")
    print("="*80)
    
    # Load original questions to test
    with open(full_dataset_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
        
    model.eval()
    
    # We will test the first 4 questions (2 forget questions, 2 retain questions)
    test_indices = [0, 1, 2, 3] 
    
    for idx in test_indices:
        item = full_data[idx]
        q = item["question"]
        gt_a = item["answer"]
        
        # Format input prompt
        prompt = f"Question: {q}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        is_forget_set = "FORGET SET" if idx < 2 else "RETAIN SET (PRESERVED)"
        print(f"\n[{is_forget_set}] Question {idx+1}: {q}")
        print(f"  -> Ground Truth: {gt_a}")
        print(f"  -> Model Output: {generated_text}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
