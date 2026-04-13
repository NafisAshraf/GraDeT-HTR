# GraDeT-HTR Training Guide

This guide provides a detailed walkthrough of the training process, dataset structure, and model management for the GraDeT-HTR project.

---

## 1. Dataset Structure
To train the model, you need a directory of images and a corresponding CSV file for labels.

### Folder Structure
```text
data/
├── sentences/
│   ├── images/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── labels.csv
├── words/
│   ├── images/
│   │   ├── word_001.png
│   │   └── ...
│   └── labels.csv
```

### Labels CSV Format (`labels.csv`)
The CSV file **must** contain at least two columns: `image_id` and `text`.
- `image_id`: The filename of the image (e.g., `img_001.png`).
- `text`: The corresponding Bengali ground truth text.

| image_id | text |
| :--- | :--- |
| img_001.png | আমাদের সোনার বাংলা |
| img_002.png | আমি তোমায় ভালোবাসি |

### How CSV Rows Map to Files in `images/`
The loader builds each image path as:

`full_path = os.path.join(images_dir, image_id)`

That means each CSV row points to exactly one file in your `images/` folder.

Example if you run with:
- `--images_dir data/sentences/images`
- `--labels_file data/sentences/labels.csv`

Then these rows map like this:

| CSV row value (`image_id`) | File that must exist |
| :--- | :--- |
| img_001.png | data/sentences/images/img_001.png |
| page12_line03.jpg | data/sentences/images/page12_line03.jpg |

### Example CSV (Sentence-Level)
```csv
image_id,text
img_001.png,আমার সোনার বাংলা
img_002.png,আমি তোমায় ভালোবাসি
img_003.png,চিরদিন তোমার আকাশ
```

### Example CSV (Word-Level)
```csv
image_id,text
word_0001.jpg,বাংলা
word_0002.jpg,ভাষা
word_0003.jpg,স্বাধীনতা
```

### Practical Rules (Important)
- `image_id` must match the actual image filename exactly (including extension, uppercase/lowercase, spaces).
- `text` should be clean ground truth text for that image.
- Keep one header row: `image_id,text`.
- Avoid empty `image_id` or empty `text` rows.
- If you accidentally keep leading/trailing spaces in `text`, those spaces become part of the label.

### Quick Validation Checklist
Before training:
- Every `image_id` in CSV exists in the selected `images_dir`.
- CSV column names are exactly `image_id` and `text`.
- You are passing matching paths (sentence CSV with sentence images, or word CSV with word images).

---

## 2. Phase 1: Sentence-Level Training
Run this phase first to teach the model general Bengali handwriting patterns.

### Command
```bash
python3 GraDeT_HTR/train.py \
    --images_dir data/sentences/images \
    --labels_file data/sentences/labels.csv \
    --checkpoint_dir checkpoints/phase1 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --num_workers 8 \
    --pin_memory \
    --persistent_workers \
    --final_model_path checkpoints/phase1/final_model_sentence.pth
```

### What happens?
- **Logs**: Training progress is printed to the terminal. You can pipe it to a file using `> phase1_training.log 2>&1`.
- **Checkpoints**: Every epoch, `checkpoints/phase1/last.pt` is updated. Periodic checkpoints (`epoch_0010.pt`) and the `best.pt` (min validation loss) are also saved.
- **History**: `checkpoints/phase1/history.json` stores loss and accuracy curves for plotting.
- **Final Model**: A standalone weights file (ready for inference) is saved to the path specified by `--final_model_path`.

---

## 3. Phase 2: Word-Level Fine-Tuning
After Phase 1 is complete, fine-tune the model on word-level images.

### Command (Resuming from Phase 1)
```bash
python3 GraDeT_HTR/train.py \
    --images_dir data/words/images \
    --labels_file data/words/labels.csv \
    --resume checkpoints/phase1/best.pt \
    --checkpoint_dir checkpoints/phase2 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.00005 \
    --final_model_path checkpoints/phase2/final_model_words.pth
```

### Why use `--resume`?
- It loads the **model weights**, **optimizer state**, and **scaler state** from the Phase 1 checkpoint.
- It resets the epoch counter if you are starting a new project, or continues if you are resuming a crashed run.

---

## 4. Understanding the Outputs

### Checkpoint Files (`.pt`)
Stored in your `--checkpoint_dir`. These are "heavy" files (containing model + optimizer state) used to **resume** training.
- `last.pt`: The state at the end of the most recent epoch.
- `best.pt`: The state that achieved the lowest validation loss so far.
- `epoch_NNNN.pt`: Snapshots saved periodically (control with `--save_every`).

### Model Weights (`.pth`)
Stored at your `--final_model_path`. These are "light" files (only model weights) used for **inference** or deployment.
- These can be loaded directly by `inference.py` or `load_final_model()` in `utils.py`.

---

## 5. Resuming a Crashed/Stopped Run
If training stops (e.g., power failure or manual stop), you can resume exactly where you left off.

```bash
python3 GraDeT_HTR/train.py \
    --images_dir data/sentences/images \
    --labels_file data/sentences/labels.csv \
    --resume checkpoints/phase1/last.pt \
    --checkpoint_dir checkpoints/phase1 \
    --epochs 50
```
*Note: Ensure the `--epochs` value is greater than the epoch number at which the model stopped.*

---

## 6. Training Tips for Large Datasets (350GB+)
- **Memory Efficiency**: Use the native `num_workers` auto-detection or set it to `8-16` depending on your CPU core count.
- **Mixed Precision**: `--use_amp` is enabled by default to speed up training and reduce GPU VRAM usage.
- **Data Split**: The code now uses an index-based split to avoid loading the entire 350GB dataset into memory twice.
- **Persistent Workers**: Use `--persistent_workers` to keep the data loading processes alive between epochs, reducing overhead.
