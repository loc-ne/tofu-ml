import subprocess
import os
import sys
# No external imports at the top to prevent ModuleNotFoundError before pip install

# Helper to run system commands with real-time printing
def run_command(command, description):
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print("="*80 + "\n")
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Read output in real-time
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    if process.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    print(f"\n[SUCCESS] Completed: {description}\n")

# Merge helper removed as full parameter unlearning does not use LoRA

def main():
    # Make sure we are in the correct directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("eval_results", exist_ok=True)
    
    # Define tasks, splits, and models
    split = "forget10"  # Predefined as 10%
    methods = {
        "GA": {"loss": "grad_ascent", "name": "grad_ascent"},
        "GD": {"loss": "grad_diff", "name": "grad_diff"},
        "KL": {"loss": "KL", "name": "KL"},
        "DPO": {"loss": "dpo", "name": "dpo"}
    }
    
    print("*"*80)
    print("   STARTING TOFU UNLEARNING REPRODUCTION PIPELINE (A100 OPTIMIZED)")
    print("*"*80)
    
    # Step 1: Install/Verify dependencies
    print("\nStep 1: Installing and verifying required packages...")
    run_command(
        "pip install -q datasets accelerate deepspeed evaluate peft rouge_score hydra-core omegaconf bitsandbytes scipy natsort matplotlib",
        "Install Pip packages"
    )
    
    # Step 2: Evaluate Retain Baseline (phi_retain90) correctly (using use_pretrained=false)
    print("\nStep 2: Evaluating Retain-90 Baseline Model...")
    run_command(
        "python evaluate_util.py model_family=phi use_pretrained=false model_path=locuslab/tofu_ft_phi-1.5_retain90 save_dir=eval_results/phi_retain90 batch_size=32 overwrite=true",
        "Evaluate Retain-90 Baseline"
    )
    
    # Step 3: Run training, evaluation, and aggregation for all 4 unlearning methods (Full Parameter)
    for key, info in methods.items():
        loss = info["loss"]
        name = info["name"]
        
        model_path = f"models/phi_unlearn_{key}"
        eval_path = f"eval_results/phi_unlearn_{key}"
        csv_path = f"eval_results/stat_{key}.csv"
        
        print("\n" + "#"*80)
        print(f" PROCESSING METHOD: {key} ({loss}) - FULL PARAMETER")
        print("#"*80 + "\n")
        
        # 3.1 Training (Full Parameter, LoRA.r=0)
        # Note: on a single A100 GPU, we run python directly without multi-GPU deepspeed.
        train_cmd = (
            f"python forget.py model_family=phi forget_loss={loss} split={split} "
            f"batch_size=4 gradient_accumulation_steps=8 lr=1e-5 num_epochs=5 "
            f"LoRA.r=0 save_model=true overwrite_dir=true save_dir={model_path}"
        )
        run_command(train_cmd, f"Train {key} Unlearning Model (Full Parameter)")
        
        # 3.2 Evaluating unlearned model
        # IMPORTANT: overwrite=true is required to force re-evaluation.
        # Without it, evaluate_util.py silently skips existing result files,
        # causing the pipeline to report metrics from a previous run's model.
        eval_cmd = (
            f"python evaluate_util.py model_family=phi model_path={model_path} "
            f"save_dir={eval_path} batch_size=32 overwrite=true"
        )
        run_command(eval_cmd, f"Evaluate {key} Unlearned Model")
        
        # 3.3 Aggregating statistics
        aggr_cmd = (
            f"python aggregate_eval_stat.py "
            f"retain_result=eval_results/phi_retain90/eval_log_aggregated.json "
            f"ckpt_result={eval_path}/eval_log_aggregated.json "
            f"method_name={name} submitted_by=Group5 save_file={csv_path}"
        )
        run_command(aggr_cmd, f"Aggregate stats for {key}")
        
    # Step 4: Plot final paper-style charts comparing all methods
    print("\n" + "#"*80)
    print(" STEP 4: GENERATING FINAL CHARTS")
    print("#"*80 + "\n")
    
    csv_files = " ".join([f"eval_results/stat_{key}.csv" for key in methods.keys()])
    plot_cmd = f"python plot_results.py {csv_files}"
    run_command(plot_cmd, "Generate Curves and Trade-off Plots")
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("Generated charts:")
    print("  - eval_results/unlearn_curves.png")
    print("  - eval_results/tradeoff_scatter.png")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
