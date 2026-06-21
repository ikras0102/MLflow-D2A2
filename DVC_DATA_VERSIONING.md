# DVC Data Versioning for D2A2

This repo now has MLflow and Optuna for experiment tracking. DVC should own the
large data directories, while MLflow records which DVC/Git data version produced
each run.

## 1. Install DVC

The current `environment.yaml` uses Python 3.7, so DVC is pinned to the last
known compatible line used here:

```bash
pip install dvc==2.10.2
```

or update the conda environment from `environment.yaml`.

## 2. Initialize DVC

Run once at the repository root:

```bash
dvc init
git add .dvc .dvcignore
```

## 3. Track Data Directories

`NYUv2/` is already ignored by Git, so it can be added directly:

```bash
dvc add NYUv2
git add NYUv2.dvc .gitignore
```

If you also want DVC to manage `MDE/`, first remove the existing tracked image
files from the Git index without deleting local files:

```bash
git rm -r --cached MDE
dvc add MDE
git add MDE.dvc .gitignore
```

Use the same pattern for test datasets such as `Lu/`, `Middlebury/`, and
`RGBDD/` if they are present locally:

```bash
dvc add Lu Middlebury RGBDD
git add Lu.dvc Middlebury.dvc RGBDD.dvc .gitignore
```

## 4. Configure Storage

For a local or shared-disk cache:

```bash
mkdir -p /path/to/dvc-store
dvc remote add -d localstore /path/to/dvc-store
dvc push
git add .dvc/config
```

For SSH/S3/OSS storage, install the matching DVC remote extra and replace the
remote URL. Keep credentials outside Git.

## 5. Train With DVC Metadata

Single train + test sketch:

```bash
python train_test_d2a2_mlflow_dvc.py \
  --dataset nyu \
  --dataset_dir NYUv2 \
  --dvc_pull \
  --dvc_required
```

Optuna HPO sketch:

```bash
python train_test_d2a2_mlflow_optuna_dvc.py \
  --dataset nyu \
  --dataset_dir NYUv2 \
  --n_trials 10 \
  --dvc_pull \
  --dvc_required
```

If a run depends on extra DVC-tracked directories, log them too:

```bash
python train_test_d2a2_mlflow_dvc.py \
  --dataset nyu \
  --dataset_dir NYUv2 \
  --dvc_data_paths MDE,pretrained \
  --dvc_pull
```

The training sketches call `dvc pull` when `--dvc_pull` is set, check that the
requested data paths exist, and log Git/DVC metadata into MLflow. The recorded
fields include `git_commit`, `git_dirty`, `dvc_dataset_path`,
`dvc_dataset_hash`, `dvc_dataset_size`, and `dvc_dataset_nfiles`.

## 6. Restore Data on Another Machine

After cloning the repo:

```bash
git pull
dvc pull
python train_test_d2a2_mlflow_dvc.py --dataset nyu --dataset_dir NYUv2
```

Use `git checkout <commit>` plus `dvc checkout` to reproduce an older committed
data pointer exactly.
