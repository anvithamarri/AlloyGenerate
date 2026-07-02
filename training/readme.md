# Fine-tuning CrystaLLM on Intermetallic Structures

## Overview
Fine-tuning was performed on Google Colaboratory using the CrystaLLM
training script on 3,661 Materials Project intermetallic structures
stored in `cifs/`.

## Requirements
- Google Colab (GPU runtime recommended — T4 or A100)
- Mount Google Drive to access `cifs/` and save checkpoint

## Usage
1. Upload `cifs/` to your Google Drive
2. Open `finetune.ipynb` in Google Colab
3. Mount Drive and set paths accordingly
4. Run all cells — training stops at ~loss 0.8

## Training details
- Base model: CrystaLLM pretrained checkpoint
- Dataset: 3,661 CIF files (cifs/)
- Final loss: ~0.81 at iteration 100,104
- Hardware: Google Colab (GPU)
- Framework: PyTorch / nanoGPT-style training loop