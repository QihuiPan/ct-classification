# Published research model cards

Updated: 2026-09-04. Maintainer: Qihui Pan.
All models are **experimental, non-clinical research artifacts**. They must not be used
for diagnosis, triage, treatment selection or patient-facing decision support.
No external-hospital, prospective or regulatory validation has been performed.
See [third-party terms and attribution](THIRD_PARTY_NOTICES.md).

## COVID-CT-MD MedicalNet fine-tune (v0.2.0)

- Trained 2026-08-31; code/results release: [v0.2.0](https://github.com/QihuiPan/ct-classification/releases/tag/v0.2.0).
- Task: non-contrast chest CT, three mutually exclusive dataset labels in order:
  `normal`, `cap`, `covid19`. CAP means community-acquired pneumonia; these labels
  do not cover other diseases or unknown cases.
- Architecture: MedicalNet 23-dataset-pretrained 3D ResNet-18, adapted to lung and
  mediastinal windows with a new three-class head.
- Data: 305 single-centre patients, split 213/46/46 train/validation/test, seed 2026.
  Class distributions, provenance and acquisition confounding are recorded in the
  [dataset card](DATASET_CARD_COVID_CT_MD.md) and [results](RESULTS_COVID_CT_MD.md).
- Selection: best validation macro AUROC, epoch 3. Train-only class weighting;
  calibration uses validation data. The three-class decision is argmax.
- Test macro AUROC **0.604**, macro AUPRC **0.444**, accuracy **0.565**.
  The model predicts every test case as COVID-19: sensitivity for CAP and normal is 0.
  An inspected Grad-CAM emphasized non-lung boundary regions. This is a failure mode,
  not evidence of useful diagnosis.
- The random-initialization baseline on the same split achieved test macro AUROC
  **0.643** and accuracy **0.609**; pretraining did not improve this test result.
- Full configuration, confidence intervals, subgroup aggregates and plots are in
  `results/covid_ct_md/medicalnet_finetune/` and `results/covid_ct_md/baseline_random_init/`.
  Sex/age analyses are exploratory and do not establish fairness.
- Fine-tune checkpoint SHA-256:
  `0cc62ed02693605f6f06078d47c0392d829fb575ad3ef2c50b178ac3a918856d`.
- Baseline checkpoint SHA-256:
  `369aed101f7c7c34d7fc4c884126632fe688778cdcb94f78879e92af32cb27df`.

## CT-RATE MedicalNet pilot (v0.3.0)

- Trained 2026-09-01 on Hugging Face Jobs; [v0.3.0](https://github.com/QihuiPan/ct-classification/releases/tag/v0.3.0).
  Training-source local Git SHA: `643789cb149cc6cf4231e6789fc83ee8e7d9d9c9`;
  equivalent published source tree at `202d8a9ba742fc7f3a544c2183af500ad06aac97`.
- Task: 18 report-derived binary abnormality labels in the exact order listed in
  [CT_RATE_PLAN.md](CT_RATE_PLAN.md). A positive output is a model score/threshold
  result, not a confirmed finding or report.
- Input: non-contrast chest CT NIfTI, two windows (lung −600/1500 HU and mediastinum
  40/400 HU), resampled spacing Z/Y/X 3.0/1.5/1.5 mm, centre crop/pad to 96×192×192.
  Body crop is not lung segmentation; field-of-view loss and acquisition shifts remain risks.
- Architecture: MedicalNet 3D ResNet-18, 18-output head, 16GB T4, AMP, batch 1,
  gradient accumulation 4; backbone frozen for the first two epochs.
- Split: 48/7/9 patients and 128/16/20 volumes (train/validation/test). Patients are
  selected by the first numbered directories, not a representative random sample.
  Official training patients stay in train; a stable patient hash partitions selected
  official validation patients into local validation and test.
- Multiple reconstructions from one scan are separate rows. Metrics and positive-label
  counts are **volume-level**, while bootstrap resamples entire patients; patients with
  more volumes contribute more weight. A volume count must not be described as a patient count.
- Loss/selection: training-only class-weighted BCE, AdamW, maximum 20 epochs, early
  stopping patience 5, best validation macro AUROC at epoch 5 of 10 actual epochs.
- Temperature calibration and Youden thresholds use validation only. Temperature
  reached the upper bound **20.0**; thresholds are unreliable with only seven patients.
- Validation macro AUROC **0.6188**; internal test macro AUROC **0.5198**, macro AUPRC
  **0.2566**, macro sensitivity **0.4345**, specificity **0.4498**, F1 **0.1420**.
  AUROC macro averaging excludes undefined labels. The existing AUPRC implementation
  contributes 0 for no-positive labels; comparisons must use the same convention.
- Test labels without positives: Medical material, Pericardial effusion,
  Peribronchial thickening, Bronchiectasis. Their AUROC is undefined. Lung nodule AUROC
  is 0.97 but thresholded sensitivity is 0; reporting the AUROC alone is misleading.
- Full per-label estimates and 200-patient-bootstrap intervals are in
  `results/ct_rate_pilot_medicalnet/test_metrics.json`. No subgroup audit was run.
- Checkpoint SHA-256:
  `b99abc58677c8947d4ac6aaf4ed25ae4999a713b814c9e7c8edf765aefc2655d`.
  Download all seven release parts and checksum manifests; reassemble using
  `python scripts/reassemble_checkpoint.py --parts-dir <download-directory>`.
  This command reads local model parts only; it never downloads CT data.

## Shared limitations and maintenance

These are frozen research releases, not deployed services. No production monitoring
or clinical maintenance SLA exists. Low-confidence flags and Grad-CAM are not validated
safety mechanisms. More epochs or more data do not guarantee acceptable performance.

Before another release, lock the test protocol, document data/label revisions, audit
acquisition and crop quality, compare baselines, assess subgroups and evaluate external
patients. Recompute calibration on a sufficiently large validation set. Publish failures
as well as improvements, and retain old artifacts for reproducibility. Suspend use if
integrity, privacy or provenance problems are found. Do not send patient data in public
issues. No differential privacy or memorization guarantee is made.

**21.3TB full CT-RATE training is not part of these completed releases.** Its budget,
cloud access and engineering prerequisites remain in [CT_RATE_PLAN.md](CT_RATE_PLAN.md).
