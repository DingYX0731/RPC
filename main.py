import os
import argparse
from huggingface_hub import hf_hub_download
import json
from build_cache import cache
from compute_perp import Evaluator as PPLEvaluator
from compute_sc import SCEvaluator
from compute_rpc import RPCEvaluator
from compute_pc import PCEvaluator

REPOID = {
    "MATH": "WNJXYK/MATH-Reasoning-Paths",
    "MathOdyssey": "WNJXYK/MathOdyssey-Reasoning-Paths",
    "AIME": "WNJXYK/AIME_1983_2024-Reasoning-Paths",
    "OlympiadBench": "WNJXYK/OlympiadBench-Reasoning-Paths"
}

EVALUATOR_MAP = {
    "PPL": PPLEvaluator,
    "SC": SCEvaluator,
    "RPC": RPCEvaluator,
    "PC": PCEvaluator
}

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, choices=["MATH", "MathOdyssey", "AIME", "OlympiadBench"], default="MathOdyssey")
args.add_argument("--model", type=str, choices=["Deepseek-Math-RL-7B", "InternLM2-Math-Plus-1.8B", "InternLM2-Math-Plus-7B"], default="InternLM2-Math-Plus-7B")
args.add_argument("--K", type=int, default=128)
args.add_argument("--method", type=str, default="PPL", choices=["PPL", "SC", "RPC", "PC"])
args.add_argument("--confidence-accuracy", action="store_true", help="Write reliability (confidence vs accuracy) bins to results.txt")
# RPC hyper-parameter robustness (Table 9 / D.6)
args.add_argument("--init-method", type=str, choices=["fixed", "zero"], default="fixed", help="RPC Weibull init: fixed or zero")
args.add_argument("--w-lower", type=float, default=0.2, help="RPC w1 lower bound")
args.add_argument("--w-upper", type=float, default=0.8, help="RPC w1 upper bound")
args.add_argument("--repeats", type=int, default=10, help="Number of random seeds (for accuracy mean ± std)")
args = args.parse_args()

repo_id = REPOID[args.dataset]
filename = args.model + ".json"

# Download sampled reasoning paths from Hugging Face
try:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(file_path, 'r', encoding='utf-8') as f:
        json_file = json.load(f)
    print(f"Load sampled reasoning paths {filename} from {repo_id} successfully!")
except Exception as e:
    print(f"Failed to load sampled reasoning paths {filename} from {repo_id}: {e}")

# Build cache for checking equality
cache_path = file_path.replace(".json", ".cache.json")
cache(json_file, cache_path)
with open(cache_path, 'r', encoding='utf-8') as f:
    cache_file = json.load(f)

# Run!
evaluator_cls = EVALUATOR_MAP[args.method]
evaluator = (
    evaluator_cls(init_method=args.init_method, w_bounds=(args.w_lower, args.w_upper))
    if args.method == "RPC"
    else evaluator_cls()
)
results = evaluator.solve(
    json_file=json_file,
    cache_file=cache_file,
    K=args.K,
    repeats=args.repeats,
    output_reliability=args.confidence_accuracy
)

# Report results
results_for_line = {k: v for k, v in results.items() if k != "reliability"}
result_str = f"{args.method} {args.dataset} {args.model} {args.K} {results_for_line}"
with open("results.txt", "a") as f:
    f.write(result_str + "\n")
    if args.confidence_accuracy and "reliability" in results:
        f.write("\nReliability (confidence vs accuracy, 10 bins):\n")
        for row in results["reliability"]:
            f.write(
                f"  Bin [{row['bin_low']:.1f}, {row['bin_high']:.1f}): "
                f"avg_confidence={row['avg_confidence']:.4f}, "
                f"accuracy%={row['accuracy_pct']:.2f}, "
                f"gap%={row['gap_pct']:.2f}, "
                f"count={row['count']:.4f}\n"
            )
        f.write("\n")
print(result_str)
if args.confidence_accuracy and "reliability" in results:
    print("Reliability bins written to results.txt")