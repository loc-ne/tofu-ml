import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_aggregated_json(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, "r") as f:
        return json.load(f)

def get_metrics_summary(eval_result_dict):
    eval_task_dict = {
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget',
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'World Facts'
    }
    
    summary = {}
    for task_file, task_name in eval_task_dict.items():
        if task_file not in eval_result_dict:
            continue
        task_data = eval_result_dict[task_file]
        
        # ROUGE
        rouge = np.array(list(task_data.get('rougeL_recall', {}).values())).mean() if 'rougeL_recall' in task_data else 0.0
        
        # Probability
        if 'eval_log' in task_file:
            gt_probs = np.exp(-1 * np.array(list(task_data.get('avg_gt_loss', {}).values())))
            prob = np.mean(gt_probs) if len(gt_probs) > 0 else 0.0
        else:
            avg_true_prob = np.exp(-1 * np.array(list(task_data.get('avg_gt_loss', {}).values())))
            avg_false_prob = np.exp(-1 * np.array(list(task_data.get('average_perturb_loss', {}).values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            prob = np.mean(avg_true_prob / avg_all_prob) if len(avg_all_prob) > 0 else 0.0
            
        # Truth Ratio
        avg_paraphrase = np.array(list(task_data.get('avg_paraphrased_loss', {}).values()))
        avg_perturbed = np.array(list(task_data.get('average_perturb_loss', {}).values())).mean(axis=-1)
        if len(avg_paraphrase) > 0 and len(avg_perturbed) > 0:
            curr_stat = np.exp(avg_perturbed - avg_paraphrase)
            if 'forget' in task_file:
                truth_ratio = np.mean(np.minimum(curr_stat, 1.0 / curr_stat))
            else:
                truth_ratio = np.mean(np.maximum(0.0, 1.0 - 1.0 / curr_stat))
        else:
            truth_ratio = 0.0
            
        summary[task_name] = {
            'ROUGE': rouge,
            'Probability': prob,
            'Truth Ratio': truth_ratio
        }
    return summary

def plot_unlearn_curves(method_summaries, save_path="unlearn_curves.png"):
    """
    Plots the performance metrics (ROUGE, Probability, Truth Ratio) 
    across 4 evaluation sets for different unlearning methods.
    method_summaries is a dict: {MethodName: summary_dict}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['ROUGE', 'Probability', 'Truth Ratio']
    eval_sets = ['Forget', 'Retain', 'Real Authors', 'World Facts']
    
    x = np.arange(len(eval_sets))
    width = 0.2
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for offset_idx, (method_name, summary) in enumerate(method_summaries.items()):
            y_vals = []
            for s in eval_sets:
                y_vals.append(summary.get(s, {}).get(metric, 0.0))
            
            ax.bar(x + (offset_idx - len(method_summaries)/2.0 + 0.5) * width, y_vals, width, label=method_name)
        
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(eval_sets)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Comparison curves saved to: {save_path}")

def plot_tradeoff_scatter(methods_data, save_path="tradeoff_scatter.png"):
    """
    Plots the trade-off between Forget Quality (Y-axis) and Model Utility (X-axis).
    methods_data is a list of dicts containing: 'name', 'model_utility', 'forget_quality'
    """
    plt.figure(figsize=(8, 6))
    
    for item in methods_data:
        plt.scatter(item['model_utility'], item['forget_quality'], s=150, label=item['name'], alpha=0.8, edgecolors='black')
        plt.text(item['model_utility'] + 0.01, item['forget_quality'] + 0.01, item['name'], fontsize=10, weight='bold')
        
    plt.xlabel('Model Utility (Higher is Better)', fontsize=12)
    plt.ylabel('Forget Quality / KS Test P-value (Higher is Better)', fontsize=12)
    plt.title('Trade-off between Model Utility and Forget Quality', fontsize=14, pad=15)
    plt.xlim(0.4, 1.0)
    plt.ylim(-0.05, 1.05)
    plt.grid(linestyle='--', alpha=0.5)
    plt.axhspan(0.05, 1.0, color='green', alpha=0.1, label='Target Forget Zone (p-value >= 0.05)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Trade-off scatter plot saved to: {save_path}")

if __name__ == "__main__":
    import sys
    import csv
    
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <path_to_csv1> [path_to_csv2] ...")
        sys.exit(1)
        
    csv_paths = sys.argv[1:]
    method_summaries = {}
    methods_data = []
    
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        with open(path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                method_name = row.get('Method', 'unknown')
                
                # Extract metrics for bar curves
                method_summaries[method_name] = {
                    'Retain': {
                        'ROUGE': float(row.get('ROUGE Retain', 0.0)),
                        'Probability': float(row.get('Prob. Retain', 0.0)),
                        'Truth Ratio': float(row.get('Truth Ratio Retain', 0.0))
                    },
                    'Forget': {
                        'ROUGE': float(row.get('ROUGE Forget', 0.0)),
                        'Probability': float(row.get('Prob. Forget', 0.0)),
                        'Truth Ratio': float(row.get('Truth Ratio Forget', 0.0))
                    },
                    'Real Authors': {
                        'ROUGE': float(row.get('ROUGE Real Authors', 0.0)),
                        'Probability': float(row.get('Prob. Real Authors', 0.0)),
                        'Truth Ratio': float(row.get('Truth Ratio Real Authors', 0.0))
                    },
                    'World Facts': {
                        'ROUGE': float(row.get('ROUGE Real World', row.get('ROUGE World Facts', 0.0))),
                        'Probability': float(row.get('Prob. Real World', row.get('Prob. World Facts', 0.0))),
                        'Truth Ratio': float(row.get('Truth Ratio Real World', row.get('Truth Ratio World Facts', 0.0)))
                    }
                }
                
                # Extract metrics for trade-off scatter
                model_utility = float(row.get('Model Utility', 0.0))
                forget_quality = float(row.get('Forget Quality', 0.0))
                methods_data.append({
                    'name': method_name,
                    'model_utility': model_utility,
                    'forget_quality': forget_quality
                })
                
    if method_summaries:
        os.makedirs("eval_results", exist_ok=True)
        plot_unlearn_curves(method_summaries, save_path="eval_results/unlearn_curves.png")
    if methods_data:
        os.makedirs("eval_results", exist_ok=True)
        plot_tradeoff_scatter(methods_data, save_path="eval_results/tradeoff_scatter.png")

