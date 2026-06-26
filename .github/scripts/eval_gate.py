"""
.github/scripts/eval_gate.py
CI model evaluation gate. Fails the build if F1 < BASELINE_F1.
"""
import os
import sys

from src.ml.train import build_features, load_data, train


def main() -> int:
    baseline = float(os.environ.get("BASELINE_F1", "0.70"))
    df = load_data()
    X, y = build_features(df)
    _, metrics = train(X, y)
    f1 = metrics["f1"]
    print(f"F1={f1:.4f} baseline={baseline}")
    if f1 < baseline:
        print("GATE FAILED")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())