# CT-RATE expanded cloud experiment

Preparation date: 2026-09-04. **Prepared, not yet trained.**
This is a larger sampled experiment, not full 21.3TB training and not clinical validation.

## Locked protocol

- Target: 512 train, 64 validation and 64 internal-test patients, exactly one volume per patient.
- Keep the official training/validation patient boundary. Within each eligible pool,
  rank patients by SHA-256 of `2026:patient_id` without looking at disease labels.
- Keep the original stable validation/test hash partition. Exclude official validation
  patients 1–16 because they were already inspected in the earlier pilot.
- Select the first scan alphabetically and lowest reconstruction index per patient.
  This avoids duplicate reconstructions but does not establish that the chosen scan is
  clinically optimal. Missing selected scans cause failure, never silent replacement.
- Use MedicalNet 3D ResNet-18, two windows, 96×192×192, batch 1, gradient accumulation 4.
  At most 20 epochs; early stopping patience 5; validation-only calibration/thresholds.
- Do not inspect test metrics while tuning. Actual disease counts and metrics will be
  published only after the cloud run; synthetic tests are not real-data validation.

## Runtime and data safeguards

Use one `t4-medium` job with a **12-hour server-enforced timeout** and no automatic retry.
The trainer has a 10-hour soft limit: it discards an incomplete epoch and evaluates the
last best complete epoch. If no complete epoch exists it reports failure, not success.
The hard timeout may still interrupt evaluation; a checkpoint alone is not a complete result.

Use the existing PyTorch 2.6.0 CUDA 12.4 image and read-only source and dataset mounts.
Set `CT_RATE_EXPANDED_CLOUD=1`, `CT_SOURCE_GIT_SHA`, and `CT_DATASET_REVISION` at submission.
Execute `bash /workspace/scripts/run_ct_rate_hf_cloud.sh`. This entrypoint refuses Windows
execution. Mount only a dedicated private output prefix at `/outputs`.

All CT reads, labels, paths, manifest and preprocessing cache remain on cloud ephemeral
storage. The 640 selected two-window tensors total about 16.9 GiB before compression;
the runner requires 40 GiB initial free space and preserves 20 GiB headroom before
reading or caching another volume. These checks are not an LRU cache or a guarantee
about the mount client's internal cache. The platform timeout remains the cost backstop.

Export only aggregate metrics, histories, resolved configuration, figures, provenance
and model weights. Patient-level artifact export is disabled. No raw CT is downloaded
to the user's computer or uploaded to GitHub.

## Launch gate

Before submission, verify the live hardware price, account balance, storage allowance,
gated dataset access, and job/output-write authorization. Reserve the job's worst-case
compute cost against the approved total budget. Record the reservation and job ID outside
the public repository; an uncertain submission must be reconciled, never blindly retried.
Do not enable ports, SSH, a schedule, extra paid storage or another job automatically.

After completion, verify the terminal job state, cohort counts, full aggregate evaluation
and checkpoint checksum before publishing a new release. Preserve earlier results;
do not relabel them as results from this expanded experiment.
