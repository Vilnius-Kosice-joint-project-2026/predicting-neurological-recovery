# Predicting Neurological Recovery

This project focuses on using Transformers and context-aware Large Language Models (LLMs) to classify patients and predict neurological recovery. It utilizes multi-modal data including EEG and other physiological signals from the I-CARE dataset.

# Conda

```powershell
conda env create -f environment.yml -n eeg
conda activate eeg
```

# Data download
```python
python download_icare.py                          # first 2 hours
python download_icare.py --hours 6                # first 6 hours
```

# Code formatting (black + isort)

This repository uses:
- `black` for consistent code formatting
- `isort` for import ordering (configured to match black)

Both tools are configured in `pyproject.toml`.

## One-time setup

```powershell
pip install black isort
```

## Format all Python files

```powershell
isort .
black .
```

## Check only (no file changes)

```powershell
isort --check-only .
black --check .
```

# Data example
```
Patient: 0984
Hospital: A
Age: 67
Sex: Male
ROSC: nan
OHCA: True
Shockable Rhythm: True
TTM: 33
Outcome: Poor
CPC: 5
```

# Run evaluation

```powershell
python analysis/official_scoring_metric/evaluate_model.py analysis/official_scoring_metric/demo_data/labels analysis/official_scoring_metric/demo_data/outputs
```