# Second dataset on Colab: NCT-CRC-HE-100K -> CRC-VAL-HE-7K

Everything big stays on Colab's scratch disk. Google Drive only gets small files
(about 3 GB at the peak, about 2 GB at the end), so a free 15 GB Drive is enough.

What runs: source ViT-B/16 fine-tuned on NCT-CRC-HE-100K (9 classes), then
frozen / TENT / Flip-TTA / EATA-LN / SHOT-IM on CRC-VAL-HE-7K, for 5 seeds.
Every step resumes if Colab disconnects.

Rough time on a T4: download 10-30 min, training 45-60 min per seed, TTA a few minutes per seed.
All five seeds fit in one 12-hour session, but split them over two sessions if you prefer.

## Step 0. Put the script in your Drive (once, about 2 minutes)

1. Open https://drive.google.com with the account that has Colab Pro (the 15 GB one).
2. Click **New** -> **New folder**, name it exactly `Next Research`, press Create.
3. Open that folder. Drag the file `crc_colab.py` into it.
   The file is on your PC at
   `E:\Temp\Next Research\02_Papers\TMI_Submission\release\colab_pro\crc_colab.py`.

You do not create any other folder. The script makes `Next Research/CRC_second_dataset/`
for its outputs by itself.

## Step 1. Open a notebook (every new session)

1. Go to https://colab.research.google.com and sign in with the same account.
2. **File** -> **New notebook**.
3. **Runtime** -> **Change runtime type**: Python 3, hardware accelerator **T4 GPU**
   (an L4 or A100 is fine if offered). Switch **High-RAM** on if the option is there. Save.
4. Keep the browser tab open while a job runs. Colab Pro does not run in the background
   when the tab is closed.

## Step 2. Setup cell (run it at the start of every session)

```python
from google.colab import drive
drive.mount('/content/drive')          # a pop-up asks for permission; click Allow
!cp "/content/drive/MyDrive/Next Research/crc_colab.py" /content/
%cd /content
!pip -q install timm scikit-learn pillow
!nvidia-smi -L
!df -h /content | tail -1
```

Check that `nvidia-smi` shows a GPU and that `df` shows at least 40 GB free.

## Step 3. Self-test (2-5 minutes, once)

```python
!python crc_colab.py smoke
```

It must end with `SMOKE OK`. If it does not, stop and open an issue with the last 30 lines.

## Step 4. Download and pack the data (every new session that trains)

```python
!python crc_colab.py prepare
```

It downloads the two zips from Zenodo into `/content/crc_work`, checks their MD5, packs them,
and deletes the big zip. It prints speed and ETA every 30 seconds. If the connection drops,
run the same cell again and it continues. `train` also calls this by itself if needed.

## Step 5. Train the source models

```python
!python crc_colab.py status
!python crc_colab.py train --seeds 1 2 3 4 5 --max-hours 10.5 --save-every 100000
```

`--max-hours 10.5` makes it stop cleanly before Colab's 12-hour limit.
`--save-every 100000` keeps only the end-of-epoch saves of the 1 GB `last.pt`. Google Drive keeps every
replaced copy in the Trash, and the Trash counts against the quota, so saving every 600 steps can fill a
free 15 GB Drive. If Drive reports `Not enough free Drive space`, empty the Trash and run the same cell again.
If Colab disconnects or stops, do Step 1 and Step 2 again, then run the same `train` line.
It resumes from the last save (at most about 6 minutes of work is lost).
Run `status` any time to see what is finished.

## Step 6. Adapt on CRC-VAL-HE-7K (two stream orders, a few minutes each)

```python
!python crc_colab.py tta --seeds 1 2 3 4 5
!python crc_colab.py tta --seeds 1 2 3 4 5 --stream sorted
```

The first line uses a fixed random order. The second uses the class-by-class order, which
matches the sorted Camelyon17 stream. Both are needed, because the sorted order changes
TENT's harm.

## Step 7. Check and pack the results

```python
!python crc_colab.py verify --seeds 1 2 3 4 5
!python crc_colab.py verify --seeds 1 2 3 4 5 --stream sorted
!python crc_colab.py pack_results
```

`verify` must say `0 failed` twice.

## Step 8. Collect the results

1. In Drive open `Next Research/CRC_second_dataset/`.
2. Right-click `CRC_results.zip` -> **Download** (a few MB; it has no checkpoints).
3. Unzip it next to this repository. The predictions that the paper uses are already in
   `crc_second_dataset/` at the repository root, and `python jbhi_revision/analyze_crc.py`
   turns them into the paper's numbers and table.

The five `best.pth` checkpoints (0.34 GB each) stay in Drive under `ckpt/`. They are not part of this
repository because of their size.

## If something goes wrong

- **`Not enough free Drive space`**: free some Drive space; nothing was written.
- **`md5 mismatch`**: the bad file is deleted; run the same cell again.
- **`Not enough local disk`**: use a runtime with more disk (Runtime -> Change runtime type).
- **CUDA out of memory**: add `--batch 32` to the `train` line (and keep the same value for `tta`).
- **Anything else**: open an issue with the last 30 lines of the cell output.
