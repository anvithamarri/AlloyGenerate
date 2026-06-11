AlloyGenerate — Intermetallic Dataset Factory
===========================================

A Streamlit app to generate, relax, validate, and export candidate intermetallic crystal structures (CIF files).

Live deployment: **https://alloygenerate.streamlit.app/**

Quick overview
--------------

- Generates CIF-like structures using a generative model and MCTS sampler.
- Relaxes structures with ASE + CHGNet and scores physical validity.
- Exports validated CIFs as a ZIP dataset and reports summary metrics.

Repo layout (important files)
-----------------------------

- `generation_model/app.py` — Streamlit entrypoint (UI + orchestration).
- `generation_model/*` — model, sampling, scoring, and helper modules.
- `cifs/` — example/generated CIF files stored in the repo.
- `generation_model/ckpt.pt` — trained model checkpoint expected at runtime.

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

If you have deployed the app, paste the public URL above where the placeholder is. Common hosting options:

- Streamlit Cloud (share.streamlit.io)
- A small VM or container (Docker) behind HTTPS

Example:

Live app: https://alloygenerate.streamlit.app/

Data and outputs
----------------

- Example CIFs and generated datasets are under the `cifs/` directory.
- Valid generated CIFs are exported as `intermetallics.zip` by the app when using the export feature.

Troubleshooting
---------------

- Ensure your virtualenv is active when running Streamlit (`which python` should point to `generation_model/venv/bin/python`).
- If CUDA/MPS is unavailable, the app will fall back to CPU (may be slower).

