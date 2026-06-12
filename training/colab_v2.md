# Colab V2 Training Notebook (cells)

**Runtime:** A100, **High-RAM** instance (≈20 GB resident for 219 features × 16 streams).

**Prereq (local):** upload BOTH zips to the Drive folder `training data`:
- `lob_training_data.zip` — fresh parquet export (`python export_training_data.py --fresh`)
- `aai_code.zip` — code snapshot: `git archive --format=zip -o aai_code.zip HEAD -- training/`
  (code ships as a zip because local `main` is unpushed; `git clone` of origin would NOT have the V2 pipeline)

**Cell 1 — mount + code**
```python
from google.colab import drive; drive.mount('/content/drive')
!rm -rf /content/aai && mkdir -p /content/aai
!unzip -q -o "/content/drive/MyDrive/training data/aai_code.zip" -d /content/aai
%cd /content/aai
# asyncpg is required even for parquet runs: training/dataset.py imports it at module top
!pip -q install torch numpy pandas pyarrow scipy asyncpg
```

**Cell 2 — GPU + RAM check**
```python
import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())
!free -g    # confirm High-RAM (~80GB)
```

**Cell 3 — pull the fresh export from Drive**
```python
!cp "/content/drive/MyDrive/training data/lob_training_data.zip" /content/aai/
!rm -rf lob_data && mkdir lob_data && unzip -q -o lob_training_data.zip -d /content/aai
!ls -la lob_data | head
```

**Cell 4 — V1 baseline (same data treatment as V2: parquet + savgol 11)**
```python
!python training/train.py --levels 40 --epochs 50 --batch-size 16 --source parquet --savgol-window 11 --run-name v1_baseline
!python training/baseline_v1.py --v1-run-dir experiments/v1_baseline --levels 40 --source parquet
# writes experiments/v1_baseline_acc.json — evaluate_v2 reads it for the V1-vs-V2 comparison
```

**Cell 5 — V2 train + eval + backtest (resumable)**
```python
!python training/colab_v2_runner.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8 --run-name v2_full_run
# If the session drops, re-run training with resume:
# !python training/train_v2.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8 --resume --source parquet --run-name v2_full_run
```

**Cell 6 — save results back to Drive**
```python
import shutil, glob, os
run_dir = 'experiments/v2_full_run' if os.path.isdir('experiments/v2_full_run') else max(glob.glob('experiments/*'), key=os.path.getmtime)
shutil.make_archive('/content/drive/MyDrive/training data/aai_v2_results', 'zip', run_dir)
shutil.copy('experiments/v1_baseline_acc.json', '/content/drive/MyDrive/training data/')
print('saved', run_dir)
```
