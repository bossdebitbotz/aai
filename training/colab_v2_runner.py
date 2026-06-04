#!/usr/bin/env python3
"""One-call runner for Colab: train V2, evaluate, backtest from a parquet export.
Assumes lob_data/ has been populated from the Drive zip and the repo is importable.

    python training/colab_v2_runner.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run(levels=40, epochs=50, batch_size=16, accum_steps=8, parquet_dir="lob_data", run_name=None):
    from training import train_v2, evaluate_v2, backtest_v2
    # 1) train (parquet source so it works on Colab without the DB)
    sys.argv = ["train_v2.py", "--levels", str(levels), "--epochs", str(epochs),
                "--batch-size", str(batch_size), "--accum-steps", str(accum_steps),
                "--source", "parquet", "--parquet-dir", parquet_dir]
    if run_name:
        sys.argv += ["--run-name", run_name]
    train_v2.main()
    # locate the run dir (most recent under experiments/)
    exp = Path(__file__).parent.parent / "experiments"
    run_dir = max(exp.iterdir(), key=lambda p: p.stat().st_mtime)
    # 2) evaluate + 3) backtest
    evaluate_v2.evaluate(str(run_dir), levels=levels, source="parquet", parquet_dir=parquet_dir)
    backtest_v2.backtest(str(run_dir), levels=levels, source="parquet", parquet_dir=parquet_dir)
    return str(run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--parquet-dir", default="lob_data")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    print("Run dir:", run(a.levels, a.epochs, a.batch_size, a.accum_steps, a.parquet_dir, a.run_name))
