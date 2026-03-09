# Workflow: Build Data -> Train SAT

1. **Build Data Phase**: Run `build_dataset.py`
   - Uses NVIDIA NIM API to extract entities and relations from `novel.json`.
   - Generates SAT-compatible graph files (`mid2id.txt`, `train.txt`, etc.).
   - Outputs files to `SAT/aligner/data/medical_kg/`.
   - *Time*: 10-30 mins depending on API rate limits and text size.

2. **Train Phase**: Run `run_sat_baseline.py` (or `main.py`)
   - The SAT model will read the newly generated `medical_kg` dataset.
   - It constructs the PyTorch Geometric graph and trains the Structure-Aware Transformer.
   - *Time*: 1-3 hours depending on GPU and dataset size.
