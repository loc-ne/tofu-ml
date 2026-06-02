from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    # Resolve relative paths using Hydra's original CWD to prevent relative path bugs
    try:
        from hydra.utils import get_original_cwd
        original_cwd = get_original_cwd()
        if cfg.save_dir is not None and not os.path.isabs(cfg.save_dir):
            cfg.save_dir = os.path.abspath(os.path.join(original_cwd, cfg.save_dir))
        if cfg.model_path is not None and not os.path.isabs(cfg.model_path):
            cfg.model_path = os.path.abspath(os.path.join(original_cwd, cfg.model_path))
        if hasattr(cfg, "eval"):
            if cfg.eval.save_dir is not None and not os.path.isabs(cfg.eval.save_dir):
                cfg.eval.save_dir = os.path.abspath(os.path.join(original_cwd, cfg.eval.save_dir))
            if cfg.eval.model_path is not None and not os.path.isabs(cfg.eval.model_path):
                cfg.eval.model_path = os.path.abspath(os.path.join(original_cwd, cfg.eval.model_path))
    except Exception:
        pass

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    local_rank = 0
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    # DeepSpeed Zero-3 is not compatible with passing a device_map, so it must be None on multi-GPU.
    # For single-GPU, we set it to "cuda" to prevent slow CPU fallback.
    device_map = "cuda" if (torch.cuda.is_available() and num_devices == 1) else None

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    # save cfg in cfg.save_dir
    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    if cfg.forget_loss == "dpo":
        torch_format_dataset = TextForgetDatasetDPOQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split)
    else:
        torch_format_dataset = TextForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit" if num_devices == 1 else "adamw_torch",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            deepspeed='config/ds_config.json' if num_devices > 1 else None,
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            eval_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed

        )
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    if os.path.exists(cfg.model_path) and os.path.isdir(cfg.model_path):
        for file in os.listdir(cfg.model_path):
            if re.search("pytorch.*\.bin", file):
                path_found = True
                break
            
            if re.search("model-*\.safetensors", file):
                path_found = True
                break
    else:
        # If it's a Hugging Face repo key (not a local path), it represents a full pretrained/finetuned model.
        # So we load it directly via AutoModelForCausalLM, which corresponds to path_found = True.
        path_found = True

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, attn_implementation="flash_attention_2" if model_cfg["flash_attention2"]=="true" else None, torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
        if cfg.forget_loss in ["KL", "dpo"]:
            oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, attn_implementation="flash_attention_2" if model_cfg["flash_attention2"]=="true" else None, torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)

    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2" if model_cfg["flash_attention2"]=="true" else None, torch_dtype=torch.bfloat16, device_map=device_map)
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true" and num_devices == 1:
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    
    import inspect
    from transformers import Trainer
    trainer_params = inspect.signature(Trainer.__init__).parameters
    
    trainer_kwargs = {
        "model": model,
        "train_dataset": torch_format_dataset,
        "eval_dataset": torch_format_dataset,
        "compute_metrics": None,
        "args": training_args,
        "data_collator": custom_data_collator_forget,
        "oracle_model": oracle_model,
        "forget_loss": cfg.forget_loss,
        "eval_cfg": cfg.eval,
    }
    
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        
    trainer = CustomTrainerForgetting(**trainer_kwargs)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        # Use trainer.save_model() which properly gathers ZeRO-3 sharded weights across all GPUs.
        # Then merge LoRA into the base model so evaluate_util.py can load without PEFT.
        print("Saving model via trainer (DeepSpeed-aware)...")
        trainer.save_model(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()

