"""
Table 9 / D.6: Performance of RPC with different initialization methods and parameter bounds.

Experiment: For each (init_method, w_bounds), run RPC with 10 random seeds (repeats=10),
report mean ± std of accuracy. No parsing of results.txt; uses evaluator.solve() directly.
"""
import re
import json
from huggingface_hub import hf_hub_download
from build_cache import cache
from compute_rpc import RPCEvaluator

REPOID = {
    "MATH": "WNJXYK/MATH-Reasoning-Paths",
    "MathOdyssey": "WNJXYK/MathOdyssey-Reasoning-Paths",
    "AIME": "WNJXYK/AIME_1983_2024-Reasoning-Paths",
    "OlympiadBench": "WNJXYK/OlympiadBench-Reasoning-Paths",
}

DATASET = "MathOdyssey"
MODEL = "InternLM2-Math-Plus-7B"
K = 128
REPEATS = 10

W_BOUNDS_LIST = [
    (0.2, 0.8),
    (0.15, 0.85),
    (0.1, 0.9),
]
INIT_METHODS = ["fixed", "zero"]


def load_data():
    repo_id = REPOID[DATASET]
    filename = MODEL + ".json"
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(file_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    cache_path = file_path.replace(".json", ".cache.json")
    cache(json_file, cache_path)
    with open(cache_path, "r", encoding="utf-8") as f:
        cache_file = json.load(f)
    return json_file, cache_file


def parse_accuracy(s):
    """Parse '31.620 ± 0.754' -> (31.620, 0.754)."""
    m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", s.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def main():
    print(f"Loading data: {DATASET} {MODEL} K={K}")
    json_file, cache_file = load_data()
    print("Running Table 9 experiments (2 inits × 3 w bounds × 10 seeds each)...\n")

    # results[w_label][init_method] = (mean, std)
    results = {}
    for (w_lo, w_hi) in W_BOUNDS_LIST:
        w_label = f"[{w_lo}, {w_hi}]"
        results[w_label] = {}
        for init_method in INIT_METHODS:
            evaluator = RPCEvaluator(init_method=init_method, w_bounds=(w_lo, w_hi))
            out = evaluator.solve(
                json_file=json_file,
                cache_file=cache_file,
                K=K,
                repeats=REPEATS,
                output_reliability=False,
            )
            mean_acc, std_acc = parse_accuracy(out["Accuracy"])
            results[w_label][init_method] = (mean_acc, std_acc)
            print(f"  w ∈ {w_label}  {init_method:10s}  ->  {out['Accuracy']}")

    # Print table (Table 9 format)
    print("\n" + "=" * 70)
    print("Table 9: Performance of RPC with different initialization methods and parameter bounds")
    print("=" * 70)
    print(f"{'w ∈ [lower, upper]':<22} | {'Fixed Init':<22} | {'Zero Init':<22}")
    print("-" * 70)
    for w_label in [f"[{w_lo}, {w_hi}]" for (w_lo, w_hi) in W_BOUNDS_LIST]:
        m_f, s_f = results[w_label]["fixed"]
        m_z, s_z = results[w_label]["zero"]
        fixed_str = f"{m_f:.3f} ± {s_f:.3f}" if m_f is not None else "N/A"
        zero_str = f"{m_z:.3f} ± {s_z:.3f}" if m_z is not None else "N/A"
        print(f"{w_label:<22} | {fixed_str:<22} | {zero_str:<22}")
    print("=" * 70)


if __name__ == "__main__":
    main()
