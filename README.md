# Dissertation: Multi-Objective Reward Modelling

Research code for training and evaluating language-model reward models. The project compares a multi-objective model scoring **safety, brevity, and coherence** with a single-reward RLHF-style baseline.

## Files

- `MORLARM.py` — trains the multi-objective reward model.
- `rlhf.py` — single-reward baseline.
- `INFERENCE.py` — generates and ranks responses using a local Ollama model.
- `calibration.py` — calibrates reward scores against human ratings.
- `EVALUATE.py` — evaluates predictions against annotated ratings.
- `TEST.py` — testing and experimentation utilities.

## Setup

Create a virtual environment and install the dependencies required by the scripts:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install torch transformers numpy scipy matplotlib seaborn tqdm requests
```

Install a PyTorch build appropriate for your hardware from the [official instructions](https://pytorch.org/get-started/locally/).

## Workflow

1. Prepare JSON training data with conversation messages and human ratings.
2. Train with `MORLARM.py` or `rlhf.py`.
3. Generate and rank responses with `INFERENCE.py`.
4. Calibrate and evaluate scores using `calibration.py` and `EVALUATE.py`.

Check each script's arguments and expected data format before running. Inference requires Ollama to be running locally.

## Note

This is academic research code. Results depend on the dataset, annotations, model configuration, and evaluation method.
