#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir \
  'SimpleITK>=2.5,<3' \
  'numpy>=1.26,<2.6' \
  'pandas>=2.2,<3.1' \
  'scikit-learn>=1.6,<2' \
  'scipy>=1.14,<2' \
  'matplotlib>=3.9,<4' \
  'seaborn>=0.13,<1' \
  'huggingface_hub>=0.28,<2' \
  'PyYAML>=6,<7' \
  'tqdm>=4.66,<5'

python -m pip install --no-cache-dir --no-deps 'monai==1.5.1'
export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
python /workspace/scripts/run_ct_rate_hf_cloud.py \
  --dataset-root /mnt/ct-rate \
  --output-root /outputs \
  --train-patients 48 \
  --valid-patients 16
