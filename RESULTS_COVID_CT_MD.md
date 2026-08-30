# COVID-CT-MD experiment results

Run date: 2026-08-31. Random seed: 2026. These are research results, not a diagnostic claim.

## Cohort and locked split

The official archive passed both checks: 11,273,767,727 bytes and MD5 `7cd2a4fdc7b1348c093b9a384d4b2240`. All 305 patient directories were decoded successfully, with 45,471 total DICOM slices and no missing paths or SeriesInstanceUID values.

| Split | Patients | Normal | CAP | COVID-19 |
|---|---:|---:|---:|---:|
| Train | 213 | 53 | 42 | 118 |
| Validation | 46 | 12 | 9 | 25 |
| Test | 46 | 11 | 9 | 26 |

The split is patient-level, stratified, deterministic, and shared by both experiments.

## Experiment comparison

| Model | Initialization | Best epoch | Validation macro-AUROC | Test macro-AUROC | Test macro-AUPRC | Test accuracy |
|---|---|---:|---:|---:|---:|---:|
| 3D ResNet-18 baseline | Random | 1 | 0.568 | 0.643 | 0.571 | 0.609 |
| 3D ResNet-18 fine-tune | MedicalNet, 23 datasets | 3 | 0.600 | 0.604 | 0.444 | 0.565 |

The MedicalNet run was selected by the validation criterion and evaluated once on the locked test split. Although pretraining improved validation macro-AUROC, it did not improve test performance. The test cohort is only 46 patients, so all estimates are unstable.

### Fine-tuned model test details

| One-vs-rest class | AUROC | Patient-bootstrap 95% CI | AUPRC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Normal | 0.652 | 0.452–0.809 | 0.329 | 0.000 | 1.000 |
| CAP | 0.634 | 0.410–0.830 | 0.438 | 0.000 | 1.000 |
| COVID-19 | 0.527 | 0.352–0.710 | 0.565 | 1.000 | 0.000 |

At the configured decision rule, the fine-tuned model predicted every test case as COVID-19. A held-out CAP inference was likewise misclassified as low-confidence COVID-19, and its Grad-CAM emphasized substantial non-lung boundary regions. The model is therefore not suitable for diagnosis or decision support.

## Reproducibility

- GPU: NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB.
- Python: 3.12.13.
- PyTorch: 2.11.0+cu128; MONAI: 1.6.0; SimpleITK: 2.5.6.
- NumPy: 2.5.2; pandas: 3.0.5; scikit-learn: 1.9.0; SciPy: 1.18.1; Matplotlib: 3.11.1.
- Fine-tuned checkpoint SHA-256: `0cc62ed02693605f6f06078d47c0392d829fb575ad3ef2c50b178ac3a918856d`.
- Random-baseline checkpoint SHA-256: `369aed101f7c7c34d7fc4c884126632fe688778cdcb94f78879e92af32cb27df`.

Aggregate JSON, histories, split counts, calibration curves, ROC/PR plots and confusion matrices are in `results/covid_ct_md/`. Raw CT data, patient-level manifests, patient-level predictions, individual CT visualizations and caches are intentionally excluded from GitHub.

## Required next work

Before any clinical interpretation, this task needs a larger multi-centre cohort, a truly external hospital test set, acquisition-protocol auditing, lung-focused preprocessing or segmentation, repeated cross-validation, and prospectively defined operating points. The current experiment is useful as a reproducible engineering baseline and as evidence that this small dataset does not support a reliable three-way classifier under the tested setup.
