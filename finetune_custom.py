import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from data_module import convert_raw_data_to_model_format

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }

class Group1Dataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
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

def main():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(BASE_DIR, "data", "group1_dataset.json")
    output_dir = os.path.join(BASE_DIR, "models", "phi_ft_group1")

    print("="*80)
    print("STARTING CUSTOM FINE-TUNING FOR GROUP 1 ON PHI-1.5")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # 1. Load Tokenizer & Model
    model_id = "microsoft/phi-1_5"
    print(f"Loading tokenizer and model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
        trust_remote_code=True
    ).to(device)

    # Hot fix for generation config
    model.generation_config.do_sample = True
    model.config.use_cache = False

    # 2. Load Dataset
    print("Loading custom dataset...")
    train_dataset = Group1Dataset(dataset_path, tokenizer, max_length=256)
    print(f"Dataset size: {len(train_dataset)} examples.")

    # 3. Training Arguments
    # Since dataset is very small (15 items), we need enough epochs (e.g., 30) for full memorization
    epochs = 20
    batch_size = 4
    learning_rate = 2e-5

    print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        save_only_model=True,
        optim="adamw_torch",
        bf16=torch.cuda.is_available(),
        seed=42,
        remove_unused_columns=False,
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_data_collator,
    )

    # 5. Train and Save
    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model and tokenizer to: {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("FINETUNING COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
