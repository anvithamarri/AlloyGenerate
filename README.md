AlloyGenerate — Intermetallic Dataset Factory
===========================================

A Streamlit app to generate, relax, validate, and export candidate intermetallic crystal structures (CIF files).

Live deployment: **https://alloygenerate.streamlit.app/**

Quick overview
--------------

- Generates CIF-like structures using a generative model and MCTS sampler.
- Relaxes structures with ASE + CHGNet and scores physical validity.
- Exports validated CIFs as a ZIP dataset and reports summary metrics.

Features
--------

- **Dynamic Device Selection**: Automatically detects and uses CUDA, MPS (Apple Silicon), or CPU.
- **Robust Path Handling**: Dynamically locates checkpoint and resources regardless of run location.
- **Material Classification**: Identifies intermetallic types (Standard, Refractory, Precious Metal).
- **Comprehensive Validation**: Checks atomic distances, energy per atom, and force thresholds.
- **Dataset Metrics**: Reports yield, validity rates, and alloy type diversity.

Repo layout (important files)
-----------------------------

- `generation_model/app.py` — Streamlit entrypoint (UI + orchestration).
- `generation_model/*.py` — model, sampling, scoring, and helper modules.
- `generation_model/requirements.txt` — Python dependencies.
- `cifs/` — finetuning CIF files.

Quickstart (local)
------------------

Prereqs: Python 3.8+, Git, and optional CUDA / Apple MPS for GPU acceleration.

1. From repository root, enter the app folder and create a virtualenv:

```bash
cd generation_model
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
source venv/bin/activate
python -m streamlit run app.py
```

3. Open the local URL that Streamlit prints (typically http://localhost:8501).

Deployment
----------

The app is deployed on Streamlit Cloud at **https://alloygenerate.streamlit.app/**

Common hosting options:
- Streamlit Cloud (share.streamlit.io)
- A small VM or container (Docker) behind HTTPS

Configuration
--------------

### Model Loading

The app uses robust path handling to locate the model checkpoint:

- The checkpoint file (`ckpt.pt`) must be in the `generation_model/` directory.
- The app dynamically discovers the directory location, making it portable across environments.

### Device Configuration

The app automatically selects the best available device:
- **CUDA** (NVIDIA GPUs) — fastest
- **MPS** (Apple Silicon) — optimized for M1/M2/M3 Macs
- **CPU** — fallback option (slower)

### Customization

In the Streamlit sidebar, you can adjust:
- **Dataset Size**: Number of generation attempts (1-5000)
- Generate button to start the process

Data and outputs
----------------

- finetuning datasets are under the `cifs/` directory.
- Valid generated CIFs are exported as `intermetallics.zip` by the app when using the export feature.
- Results include comprehensive metrics:
  - Valid crystal count and yield percentages
  - Energy per atom (eV)
  - Maximum atomic forces (eV/Å)
  - Alloy type distribution

Dependencies
------------

Key dependencies (see `requirements.txt` for complete list):
- **Streamlit** — UI framework
- **PyTorch** — model inference
- **ASE** — atomic structure handling
- **CHGNet** — structure relaxation and energy calculations
- **PyMatGen** — materials informatics
