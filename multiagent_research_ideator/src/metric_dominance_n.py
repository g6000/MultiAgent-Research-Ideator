import argparse
import json
import random


def compute_dominance(scores, N_values):
    # Sort by score in descending order
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Classify into A and B groups
    results = {n: {"A": 0, "B": 0} for n in N_values}

    for N in N_values:
        top_n = sorted_items[:N] if N != "all" else sorted_items
        a_count = sum(1 for k, _ in top_n if k.startswith("A_"))
        b_count = sum(1 for k, _ in top_n if k.startswith("B_"))

        results[N]["A"] = a_count / len(top_n)
        results[N]["B"] = b_count / len(top_n)

    return results


def print_dominance(results):
    print(f"{'@N':<8}{'A Score':<10}{'B Score':<10}")
    print("-" * 28)
    for N in results:
        a_score = results[N]["A"]
        b_score = results[N]["B"]
        label = str(N).rjust(3)
        print(f"{label:<8}{a_score:<10.2f}{b_score:<10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to JSON file containing method scores")
    parser.add_argument(
        "--seed", type=int, default=2024, help="Seed for random number generator"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.json_file, "r") as f:
        scores = json.load(f)

    total = len(scores)
    N_values = [5, 10, 20, 40]
    print("Number of evaluation cases: " + str(total))
    if total > 0:
        N_values.append(f"all")

    dominance_results = compute_dominance(scores, N_values)
    print_dominance(dominance_results)
