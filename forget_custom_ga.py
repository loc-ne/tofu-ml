import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from data_module import convert_raw_data_to_model_format

class ForgetDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=256):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = {
            'question_start_tag': 'Question: ',
            'question_end_tag': '\n',
            'answer_tag': 'Answer: '
        }

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
    forget_dataset_path = os.path.join(BASE_DIR, "data", "forget_group1.json")
    full_dataset_path = os.path.join(BASE_DIR, "data", "group1_dataset.json")
    output_dir = os.path.join(BASE_DIR, "models", "phi_unlearn_GA_group1")

    print("="*80)
    print("STARTING CUSTOM GRADIENT ASCENT UNLEARNING FOR GROUP 1 ON PHI-1.5")
    print(f"Base model directory: {model_dir}")
    print(f"Forget dataset path: {forget_dataset_path}")
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

    # 2. Load Forget Dataset
    forget_dataset = ForgetDataset(forget_dataset_path, tokenizer)
    dataloader = DataLoader(forget_dataset, batch_size=2, shuffle=True, collate_fn=custom_data_collator)
    print(f"Forget dataset size: {len(forget_dataset)} examples.")

    # 3. Setup Optimizer for Gradient Ascent
    # GA uses a very small learning rate to prevent catastrophic forgetting of other facts
    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    epochs = 5

    print(f"Unlearning configuration: epochs={epochs}, learning_rate={lr}")
    print("Running Gradient Ascent unlearning loop...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            
            # Gradient Ascent Loss: minimize the negative likelihood (maximize the loss)
            # We multiply outputs.loss by -1
            loss = outputs.loss * -1.0
            
            loss.backward()
            optimizer.step()
            
            # We track the absolute value of the original loss to print it
            epoch_loss += outputs.loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Original Loss (Confidence): {epoch_loss/len(dataloader):.4f}")

    # 4. Save the Unlearned Model
    print(f"Saving unlearned model to: {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[SUCCESS] Unlearning completed and model saved.")

    # 5. Live Demo / Verification Block
    print("\n" + "="*80)
    print("   LIVE DEMO: VERIFYING BEFORE VS AFTER UNLEARNING")
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
        
        is_forget_set = "FORGET SET" if idx < 2 else "RETAIN SET"
        print(f"\n[{is_forget_set}] Question {idx+1}: {q}")
        print(f"  -> Ground Truth: {gt_a}")
        print(f"  -> Model Output: {generated_text}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
