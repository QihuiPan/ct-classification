# COVID-CT-MD dataset card

## Intended experiment

- Patient-level, three-way classification of non-contrast chest CT volumes.
- Stable labels used by this repository: `normal`, `cap`, `covid19`.
- Research baseline only; it is not a diagnostic device or a clinically validated model.

## Published cohort

| Repository label | Clinical category | Patients | Published sex distribution | Published mean age |
|---|---|---:|---|---:|
| `covid19` | rRT-PCR-positive COVID-19 | 169 | 108 M / 61 F | 51.96 ± 14.39 |
| `cap` | Community-acquired pneumonia | 60 | 35 M / 25 F | 57.7 ± 21.7 |
| `normal` | Normal CT cohort | 76 | 40 M / 36 F | 43.4 ± 14.1 |
| **Total** |  | **305** | **183 M / 122 F** | — |

Sources: the [official dataset repository](https://github.com/ShahinSHH/COVID-CT-MD), the [Figshare collection](https://figshare.com/collections/5129081), and the [Scientific Data article](https://doi.org/10.1038/s41597-021-00900-3).

## Local E-drive layout

```text
E:/Codex/ct-classification/
├── datasets/COVID-CT-MD/
│   ├── COVID-CT-MD.zip
│   ├── raw/
│   └── manifest.csv
├── cache/covid_ct_md/
├── runs/covid_ct_md/
├── pip-cache/
├── tmp/
└── venv/
```

The raw CT data, manifest, preprocessing cache, virtual environment, model checkpoint and detailed predictions remain on E:. The GitHub repository contains only reusable code, configuration, documentation, aggregate metrics, plots, and any release artifacts that are safe and permitted to publish.

## Integrity checks

- Official archive filename: `COVID-CT-MD.zip`
- Expected byte size: `11,273,767,727`
- Expected MD5: `7cd2a4fdc7b1348c093b9a384d4b2240`
- `scripts/prepare_covid_ct_md.py` refuses the full-dataset manifest unless it discovers exactly 169 COVID-19, 60 CAP, and 76 normal patients.
- The training split is patient-level and stratified; the random seed is fixed at 2026.

## Known limitations

- This is a small, single-centre cohort. Internal random-split performance cannot establish transportability to another hospital, scanner, reconstruction protocol, geography, or time period.
- Collection periods differ: COVID-19 cases were collected in February–April 2020, while CAP and normal cohorts include earlier dates. A model may learn temporal or acquisition differences instead of pathology.
- The classes are imbalanced. The configuration uses training-only class weights and reports per-class as well as macro metrics.
- Age and sex subgroup results are exploratory because several subgroups are small.
- A high test score on this dataset must not be interpreted as clinical readiness.
