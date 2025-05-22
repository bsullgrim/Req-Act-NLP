import pandas as pd
import re
import pandas.errors

def evaluator_with_prf(manual_file="manual_matches.csv", auto_file="hybrid_matches_trf.csv", top_n=5):
    try:
        auto_df = pd.read_csv(auto_file)
    except pandas.errors.EmptyDataError:
        print(f"[Evaluator] Skipping evaluation: {auto_file} is empty or has no parseable columns.")
        return {
            "total": 0,
            "top_n_accuracy": 0.0,
            "top_1_accuracy": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "mean_mrr": 0.0,
        }, pd.DataFrame()

    def normalize_activity_name(name):
        return re.sub(r'\(context.*?\)', '', name, flags=re.IGNORECASE).strip().lower()

    manual_df = pd.read_csv(manual_file)

    grouped_auto = auto_df.groupby("ID", group_keys=False).apply(
        lambda x: list(x.drop(columns=["ID"]).sort_values("Combined Score", ascending=False)["Activity Name"].head(top_n))
    )

    merged = manual_df.set_index("ID").join(grouped_auto.rename("Top Matches"))

    def normalize_list(names):
        return [normalize_activity_name(n) for n in names if n]

    def matched_in_top(row):
        try:
            manual = normalize_list(row["Satisfied By"].split(","))
            predicted = normalize_list(row["Top Matches"])
            return any(m in predicted for m in manual)
        except Exception:
            return False

    merged["Matched in Top N"] = merged.apply(matched_in_top, axis=1)

    def top1_match(row):
        try:
            manual = normalize_list(row["Satisfied By"].split(","))
            predicted = normalize_list(row["Top Matches"])
            if not manual or not predicted:
                return False
            return manual[0] == predicted[0]
        except Exception:
            return False

    merged["Top1 Match"] = merged.apply(top1_match, axis=1)

    def precision_recall_f1(row):
        try:
            manual_set = set(normalize_list(row["Satisfied By"].split(",")))
            predicted_set = set(normalize_list(row["Top Matches"]))
            tp = len(manual_set.intersection(predicted_set))
            precision = tp / len(predicted_set) if predicted_set else 0
            recall = tp / len(manual_set) if manual_set else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return pd.Series({"Precision": precision, "Recall": recall, "F1": f1})
        except Exception:
            return pd.Series({"Precision": 0, "Recall": 0, "F1": 0})

    prf = merged.apply(precision_recall_f1, axis=1)
    merged = pd.concat([merged, prf], axis=1)

    def reciprocal_rank(row):
        try:
            manual = normalize_list(row["Satisfied By"].split(","))
            predicted = normalize_list(row["Top Matches"])
            for rank, pred in enumerate(predicted, start=1):
                if pred in manual:
                    return 1 / rank
            return 0
        except Exception:
            return 0

    merged["Reciprocal Rank"] = merged.apply(reciprocal_rank, axis=1)

    total = len(merged)
    top_n_hits = merged["Matched in Top N"].sum()
    top_1_hits = merged["Top1 Match"].sum()
    mean_precision = merged["Precision"].mean()
    mean_recall = merged["Recall"].mean()
    mean_f1 = merged["F1"].mean()
    mean_mrr = merged["Reciprocal Rank"].mean()

    summary = {
        "total": total,
        "top_n_accuracy": top_n_hits / total if total > 0 else 0,
        "top_1_accuracy": top_1_hits / total if total > 0 else 0,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "mean_mrr": mean_mrr,
    }

    print(f"Total requirements evaluated: {total}")
    print(f"Top-{top_n} accuracy: {top_n_hits / total:.2%}")
    print(f"Top-1 accuracy: {top_1_hits / total:.2%}")
    print(f"Mean Precision@{top_n}: {mean_precision:.2%}")
    print(f"Mean Recall@{top_n}: {mean_recall:.2%}")
    print(f"Mean F1@{top_n}: {mean_f1:.2%}")
    print(f"Mean Reciprocal Rank (MRR): {mean_mrr:.3f}")

    missed = merged[~merged["Matched in Top N"]][["Satisfied By", "Top Matches"]]
    if not missed.empty:
        print("\nMissed Matches (up to 10 examples):")
        print(missed.head(10))

    return summary, merged

if __name__ == "__main__":
    evaluator_with_prf()