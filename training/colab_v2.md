# Colab V2 Training Notebook (cells)

**Runtime:** A100, **High-RAM** instance (≈20 GB resident for 219 features × 16 streams).

**Cell 1 — mount + repo**
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <your-repo-url> /content/aai || (cd /content/aai && git pull)
%cd /content/aai
!pip -q install torch numpy pandas pyarrow scipy
```

**Cell 2 — GPU + RAM check**
```python
import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())
!free -g    # confirm High-RAM (~80GB)
```

**Cell 3 — pull the fresh export from Drive**
```python
!cp "/content/drive/MyDrive/training data/lob_training_data.zip" /content/aai/
!rm -rf lob_data && mkdir lob_data && unzip -o lob_training_data.zip -d /content/aai
!ls -la lob_data | head
```

**Cell 4 — train + eval + backtest (resumable)**
```python
!python training/colab_v2_runner.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8 --run-name v2_full_run
# If the session drops, re-run training with resume:
# !python training/train_v2.py --levels 40 --epochs 50 --batch-size 16 --accum-steps 8 --resume --source parquet --run-name v2_full_run
```

**Cell 5 — save results back to Drive**
```python
import shutil, glob, os
run_dir = max(glob.glob('experiments/*'), key=os.path.getmtime)
shutil.make_archive('/content/drive/MyDrive/training data/aai_v2_results', 'zip', run_dir)
print('saved', run_dir)
```
