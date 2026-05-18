# NeuroScan — EEG Disorder Detection

Deep learning pipeline and interactive demo for classifying EEG signals. The project trains **CNN**, **BiLSTM**, and **Hybrid (CNN + attention + BiLSTM)** models on the [BONN University EEG dataset](https://www.upr-si.net/en/research/resources-for-researchers/research-data/) and serves predictions through a **Streamlit** app called **NeuroScan**.

> **Disclaimer:** This tool is for research and education only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

---

## Features

- **Two classification tasks** on BONN data:
  - **Binary:** Normal vs seizure (ictal)
  - **5-class:** Z, O, N, F, S recording groups (healthy, eyes-open healthy, interictal, focal interictal, ictal)
- **Three model architectures** trained and compared side by side
- **Preprocessing:** band-pass filtering (0.5–45 Hz), z-score normalization, overlapping 512-sample segments
- **Web UI:** upload `.txt` / `.csv` EEG files (plus optional `.edf` / `.set` via MNE), view waveforms, frequency bands, and ensemble-style predictions from all models
- **Metrics & curves:** validation accuracy, precision, recall, F1, and per-epoch training history saved as JSON

---

## Results (validation set)

| Task | CNN | BiLSTM | Hybrid |
|------|-----|--------|--------|
| BONN Binary | 99.40% | 99.47% | **99.60%** |
| BONN 5-Class | 83.60% | 82.00% | **86.60%** |

Metrics are read from `saved_models/*_meta.json` after training. Your numbers may vary slightly if you retrain.

---

## Project structure

```
ML_DL/
├── 01_setup_and_data.py      # Load BONN, preprocess, save .npy arrays
├── 06_train_and_save_models.py  # Train CNN / BiLSTM / Hybrid, save checkpoints
├── app.py                    # Streamlit UI (NeuroScan)
├── helper.py                   # Utility stub
├── data/
│   ├── bonn/                 # Raw BONN .txt files (Z, O, N, F, S folders)
│   ├── processed/            # bonn_X.npy, bonn_y2.npy, bonn_y5.npy
│   ├── adftd/                # Optional ADFTD / Alzheimer’s EEG (BIDS-style)
│   └── uci/                  # Additional EEG datasets
├── saved_models/             # *.pt weights + *_meta.json
├── results/                  # Training history JSON + plots
└── README.md
```

---

## Requirements

- Python 3.9+
- macOS with Apple Silicon: training uses **MPS** when available; otherwise **CPU**
- Dependencies:

```bash
pip install numpy scipy matplotlib scikit-learn torch streamlit pandas
# Optional — for .edf / .set uploads in the app:
pip install mne
```

---

## Quick start

### 1. Clone and enter the repo

```bash
git clone git@github.com:Athina09/ML-DL-project.git
cd ML-DL-project
```

### 2. Add the BONN dataset

Download the five-class BONN EEG seizure corpus and place the recording folders under `data/bonn/`:

```
data/bonn/
├── Z/   # healthy, eyes closed
├── O/   # healthy, eyes open
├── N/   # interictal (non-seizure)
├── F/   # interictal (focal)
└── S/   # ictal (seizure)
```

Each folder should contain one `.txt` file per recording (4097 samples per file at 173 Hz sampling rate, as used in this project).

### 3. Preprocess data

```bash
python3 01_setup_and_data.py
```

This writes:

- `data/processed/bonn_X.npy` — segmented signals (512 samples, 50% overlap)
- `data/processed/bonn_y5.npy` — 5-class labels
- `data/processed/bonn_y2.npy` — binary labels (seizure vs non-seizure)
- `results/sample_signals.png` — example plots

### 4. Train models

```bash
python3 06_train_and_save_models.py
```

Training runs both tasks sequentially (~tens of minutes on GPU/MPS). Outputs:

- `saved_models/{cnn,bilstm,hybrid}_{bonn_binary,bonn_5class}.pt`
- `saved_models/{bonn_binary,bonn_5class}_meta.json`
- `results/{model}_{task}_history.json`

### 5. Launch the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

---

## Using NeuroScan

1. In the sidebar, choose **BONN Binary** or **BONN 5-Class**.
2. Upload an EEG file:
   - **BONN format:** plain `.txt` with one sample per line (try files from `data/bonn/S/` for seizure, `data/bonn/Z/` for normal).
   - **Other formats:** `.csv`, or `.edf` / `.set` if MNE is installed.
3. Toggle **CNN**, **BiLSTM**, and **Hybrid** to compare predictions.
4. Use **Model Comparison** and **Training History** tabs to inspect saved metrics and learning curves.

If model files are missing, the app falls back to **demo mode** with random probabilities until training completes.

---

## Model architectures

| Model | Description |
|-------|-------------|
| **CNN** | 1D convolutions (32→64→128 filters), global average pooling, dense classifier |
| **BiLSTM** | Two stacked bidirectional LSTM layers with batch norm |
| **Hybrid** | CNN feature extractor → multi-head self-attention → BiLSTM → classifier |

All models use **512-sample** input windows, **Adam** optimizer, **class-weighted cross-entropy**, learning-rate scheduling, and **early stopping** (patience 10 on validation loss).

---

## BONN class mapping

| Folder | Label | Meaning |
|--------|-------|---------|
| Z | 0 | Healthy (eyes closed) |
| O | 1 | Healthy (eyes open) |
| N | 2 | Seizure-free (interictal) |
| F | 3 | Seizure-free (focal interictal) |
| S | 4 | Ictal (seizure) |

Binary task: label **4 (S)** → seizure; all others → normal.

---

## Optional data

- **`data/adftd/`** — ADFTD Alzheimer’s EEG (BIDS layout); referenced in the UI for future extension.
- **`data/uci/`** — UCI EEG Eye State and related files.

These are not required for the default BONN training workflow.

---

## License & attribution

- **Code:** see repository license (if added).
- **BONN University EEG data:** cite the original dataset authors when publishing work based on this project. See the [UPR research data page](https://www.upr-si.net/en/research/resources-for-researchers/research-data/) for terms of use.

---

## Repository

https://github.com/Athina09/ML-DL-project
