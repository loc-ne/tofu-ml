import subprocess
import os
import sys
from datetime import datetime

# Helper to run system commands with real-time printing
def run_command(command, description):
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print("="*80 + "\n")

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end="")

    process.wait()
    if process.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    print(f"\n[SUCCESS] Completed: {description}\n")


def main():
    # CRITICAL: Capture absolute working directory BEFORE any subprocess runs.
    # Hydra (used by forget.py, evaluate_util.py, aggregate_eval_stat.py) changes
    # the working directory to outputs/YYYY-MM-DD/HH-MM-SS/ inside each subprocess.
    # Without absolute paths, each script resolves relative paths to DIFFERENT locations
    # → model saved to one place, eval reads from another → always evaluates old model!
    BASE_DIR = os.path.abspath(os.getcwd())
    print(f"Base directory: {BASE_DIR}")

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "eval_results"), exist_ok=True)

    split = "forget10"
    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run tag: {RUN_TAG}")

    methods = {
        "GA":  {"loss": "grad_ascent", "name": "grad_ascent"},
        "GD":  {"loss": "grad_diff",   "name": "grad_diff"},
        "KL":  {"loss": "KL",          "name": "KL"},
        "DPO": {"loss": "dpo",         "name": "dpo"},
    }

    print("*"*80)
    print("   STARTING TOFU UNLEARNING REPRODUCTION PIPELINE (A100 OPTIMIZED)")
    print("*"*80)

    # Step 1: Install dependencies
    print("\nStep 1: Installing required packages...")
    run_command(
        "pip install -q datasets accelerate deepspeed evaluate peft rouge_score "
        "hydra-core omegaconf bitsandbytes scipy natsort matplotlib",
        "Install Pip packages"
    )

    # Step 2: Verify retain-90 baseline (authors pre-computed, stored in data/)
    # README: "The retain results are uploaded in data/"
    # ABSOLUTE path required: aggregate_eval_stat.py runs under Hydra (different cwd).
    RETAIN90_RESULT = os.path.join(
        BASE_DIR, "data",
        "ft_epoch5_lr2e-05_phi_retain90_wd0.01",
        "eval_results", "ds_size300", "eval_log_aggregated.json"
    )
    print(f"\nStep 2: Retain-90 baseline: {RETAIN90_RESULT}")
    if not os.path.exists(RETAIN90_RESULT):
        print(f"[ERROR] Retain-90 result file not found: {RETAIN90_RESULT}")
        sys.exit(1)
    print("[OK] Retain-90 baseline found.")

    # Step 3: Train → Evaluate → Aggregate for each unlearning method
    for key, info in methods.items():
        loss = info["loss"]
        name = info["name"]
        lr   = info["lr"]

        # ALL paths are ABSOLUTE to survive Hydra's cwd change inside subprocesses
        model_path = os.path.join(BASE_DIR, "models", f"phi_unlearn_{key}")
        eval_path  = os.path.join(BASE_DIR, "eval_results", f"phi_unlearn_{key}_{RUN_TAG}")
        csv_path   = os.path.join(BASE_DIR, "eval_results", f"stat_{key}_{RUN_TAG}.csv")

        print("\n" + "#"*80)
        print(f" PROCESSING METHOD: {key} ({loss}, lr={lr}) - FULL PARAMETER")
        print("#"*80 + "\n")

        # 3.1 Train (Full Parameter, LoRA off)
        # forget.py loads ft_model_path="locuslab/tofu_ft_phi-1.5" (public HF) by default.
        # save_dir is ABSOLUTE so Hydra's cwd change doesn't redirect the saved model.
        train_cmd = (
            f"python forget.py model_family=phi forget_loss={loss} split={split} "
            f"batch_size=4 gradient_accumulation_steps=8 lr={lr} num_epochs=5 "
            f"LoRA.r=0 save_model=true overwrite_dir=true save_dir={model_path}"
        )
        run_command(train_cmd, f"Train {key} - Full Parameter Unlearning (lr={lr})")

        # 3.2 Evaluate unlearned model
        # model_path and save_dir are ABSOLUTE to survive Hydra's cwd change.
        # overwrite=true forces fresh evaluation (not skipping existing JSON files).
        eval_cmd = (
            f"python evaluate_util.py model_family=phi model_path={model_path} "
            f"save_dir={eval_path} batch_size=32 overwrite=true"
        )
        run_command(eval_cmd, f"Evaluate {key} Unlearned Model")

        # 3.3 Aggregate statistics (KS-test for Forget Quality + Model Utility)
        # All three file paths are ABSOLUTE.
        aggr_cmd = (
            f"python aggregate_eval_stat.py "
            f"retain_result={RETAIN90_RESULT} "
            f"ckpt_result={eval_path}/eval_log_aggregated.json "
            f"method_name={name} submitted_by=Group5 save_file={csv_path}"
        )
        run_command(aggr_cmd, f"Aggregate stats for {key}")

    # Step 4: Plot comparison charts
    print("\n" + "#"*80)
    print(" STEP 4: GENERATING FINAL CHARTS")
    print("#"*80 + "\n")

    csv_files = " ".join([
        os.path.join(BASE_DIR, "eval_results", f"stat_{key}_{RUN_TAG}.csv")
        for key in methods.keys()
    ])
    run_command(f"python plot_results.py {csv_files}", "Generate Comparison Charts")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  Run tag : {RUN_TAG}")
    print(f"  CSVs    : eval_results/stat_*_{RUN_TAG}.csv")
    print(f"  Charts  : eval_results/unlearn_curves.png")
    print(f"            eval_results/tradeoff_scatter.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
