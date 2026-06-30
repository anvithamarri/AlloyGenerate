# Model Checkpoint — `ckpt.pt`

## Status

This repository uses **Git LFS** (Large File Storage) to track `ckpt.pt`, since the trained
CrystaLLM checkpoint is approximately **296 MB** — too large for a standard Git object.

If you cloned this repository normally without Git LFS installed, the file at
`generation_model/ckpt.pt` will **not** be the actual model weights. Instead, it will be a
small (~130 byte) text pointer file that looks like this:
version https://git-lfs.github.com/spec/v1
oid sha256:a224e8d2ecea0163610504e55463735b3e0c1e8dfebc36ac9ce78f502aba76a8
size 310723569

If `app.py` or any validation script fails with an error related to loading `ckpt.pt`
(e.g., `RuntimeError`, `UnpicklingError`, or a checkpoint size/shape mismatch), this is
almost certainly the cause — check the file size first using the verification steps below.

## How to get the real checkpoint file

### Step 1 — Install Git LFS (one-time, per machine)

**macOS (Homebrew):**
```bash
brew install git-lfs
```

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install git-lfs
```

**Windows:**
Download and run the installer from [git-lfs.com](https://git-lfs.com).

**Conda (cross-platform alternative):**
```bash
conda install -c conda-forge git-lfs
```

### Step 2 — Initialize Git LFS for your user account (one-time)

```bash
git lfs install
```

You should see confirmation output similar to:
Git LFS initialized.

### Step 3 — Get the file

**If you haven't cloned the repository yet:**
Git LFS files are fetched automatically as part of a normal clone, once Step 1 and Step 2
are complete:

```bash
git clone https://github.com/anvithamarri/AlloyGenerate.git
```

**If you already cloned the repository before installing Git LFS:**
Pull the actual file content after installing LFS:

```bash
cd AlloyGenerate
git lfs pull
```

### Step 4 — Verify the file is correct

```bash
ls -lh generation_model/ckpt.pt
```

Expected output: a file size around **296M** (not a few hundred bytes).

```bash
file generation_model/ckpt.pt
```

Expected output: a binary data indicator (e.g., `data`), **not** `ASCII text`. If it still
says `ASCII text`, the pointer file was not replaced — `git lfs pull` did not complete
successfully; re-run Step 3.

**Confirm checksum (recommended before first use):**
```bash
sha256sum generation_model/ckpt.pt
```
Expected: `a224e8d2ecea0163610504e55463735b3e0c1e8dfebc36ac9ce78f502aba76a8`

(On macOS, use `shasum -a 256 generation_model/ckpt.pt` instead.)

## File details

| Property | Value |
|---|---|
| File size | 310,723,569 bytes (~296 MB) |
| SHA-256 | `a224e8d2ecea0163610504e55463735b3e0c1e8dfebc36ac9ce78f502aba76a8` |
| Format | PyTorch checkpoint (`torch.save` dict, keys: `model`, `model_args`) |
| Architecture | GPT-style autoregressive transformer (see `model_utils.py` for `GPTConfig`) |

## Troubleshooting

**`git lfs pull` does nothing, or `git lfs` command not found**
Git LFS isn't installed, or `git lfs install` was never run on this machine. Repeat
Step 1 and Step 2 above, then retry Step 3.

**`git lfs pull` fails with a "smudge filter" or authentication error**
Run `git lfs install --force` to re-register the LFS hooks, then retry:
```bash
git lfs install --force
git lfs pull
```

**`git lfs pull` fails with a bandwidth or quota exceeded error**
GitHub's free-tier Git LFS bandwidth has a monthly cap shared across all repository
clones and downloads. This can occur on heavily-cloned public repositories. If this
happens repeatedly, wait for the quota to reset (monthly) or contact the repository
maintainer.

**File size looks correct (~296 MB) but `app.py` still fails to load it**
Confirm the SHA-256 checksum (Step 4 above). A partial or interrupted LFS download can
sometimes produce a file of approximately the right size that is still corrupted.

**`git clone` is very slow or times out**
This is typically normal for a ~296 MB LFS download on a slow connection, not an error.
Allow extra time, or run `git lfs pull` separately after a faster shallow clone:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/anvithamarri/AlloyGenerate.git
cd AlloyGenerate
git lfs pull
```