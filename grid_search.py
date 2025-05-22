import itertools
import pandas as pd
from matcher import run_matcher  # assuming your matcher script is saved as matcher.py
from evaluator import evaluator_with_prf  # assuming your evaluator is saved as evaluator.py

def grid_search():
    vn_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sem_weights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]  # complementary to vn_weights
    min_sims = [0.2, 0.3, 0.4, 0.5, 0.6]
    top_ns = [5]

    results = []

    for vn, sem, min_sim, top_n in itertools.product(vn_weights, sem_weights, min_sims, top_ns):
        print(f"\nRunning matcher with vn_weight={vn}, sem_weight={sem}, min_sim={min_sim}, top_n={top_n}")
        out_file = f"hybrid_matches_trf_vn{vn}_sem{sem}_min{min_sim}_top{top_n}"
        run_matcher(vn_weight=vn, sem_weight=sem, min_sim=min_sim, top_n=top_n, out_file=out_file)

        # Evaluate on produced CSV and get summary metrics dict
        metrics, _ = evaluator_with_prf(manual_file="manual_matches.csv", auto_file=out_file + ".csv", top_n=top_n)

        if metrics is None:
            print(f"[Warning] Skipping: No metrics returned for {out_file}. Possibly empty result file.")
            continue

        # Add hyperparameters to the metrics dict
        metrics["vn_weight"] = vn
        metrics["sem_weight"] = sem
        metrics["min_sim"] = min_sim
        metrics["top_n"] = top_n
        results.append(metrics)

    # Save results only if we have any
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("grid_search_results.csv", index=False)
        print("\nGrid search complete. Results saved to grid_search_results.csv")
    else:
        print("\nGrid search complete. No valid results to save.")

if __name__ == "__main__":
    grid_search()