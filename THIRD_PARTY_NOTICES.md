# Data, pretrained models and publication boundaries

Public visibility does not make the medical datasets public-domain, grant clinical approval,
or apply a blanket MIT license to this project. No new license for the original project
code is selected here; upstream rights and conditions remain in force.

## CT-RATE

Source: [Ibrahim Ethem Hamamci and collaborators, CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).
The dataset card identifies **CC BY-NC-SA 4.0**, with additional gated-access conditions.
The gate prohibits redistribution of any portion of the dataset and re-identification;
derived material must preserve privacy. Obtain data directly from the authors and accept
their conditions. Do not upload CT volumes, reports, label tables, metadata, manifests,
individual predictions or credentials to this repository.

The v0.3.0 checkpoint is a modified MedicalNet model fine-tuned on a small CT-RATE subset,
not a copy of the dataset. CT-RATE-derived research artifacts are provided for
non-commercial research under the applicable [CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/),
with attribution to the data authors and preservation of upstream notices. No ownership
of the dataset is claimed. The checkpoint has not undergone a formal membership-inference
or memorization audit; absence of raw patient records is not a differential-privacy guarantee.

Dataset citation: Hamamci et al., *Generalist foundation models from a multimodal dataset
for 3D computed tomography*, Nature Biomedical Engineering (2026).
See the [authors' paper and citation record](https://arxiv.org/abs/2403.17834).

## COVID-CT-MD

Source: [Afshar et al., COVID-CT-MD](https://github.com/ShahinSHH/COVID-CT-MD),
[official Figshare collection](https://figshare.com/collections/5129081),
[Scientific Data paper](https://doi.org/10.1038/s41597-021-00900-3).
Download data from the official source under its current license; no dataset or
patient-level artifacts are redistributed here. The paper's license must not be assumed
to grant different rights over individual dataset files. Models and evaluations are
experimental research artifacts, not diagnostic products.

## MedicalNet pretrained weights

Source: [Tencent MedicalNet](https://github.com/Tencent/MedicalNet),
[official ResNet-18 weights](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18).
Citation: Chen, Ma and Zheng, *Med3D: Transfer Learning for 3D Medical Image Analysis* (2019),
[arXiv:1904.00625](https://arxiv.org/abs/1904.00625).
The fine-tuned models adapt the first convolution from one channel to two CT windows
and replace the classification head. The following upstream notice is retained from
[MedicalNet/LICENSE](https://github.com/Tencent/MedicalNet/blob/master/LICENSE).

Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.

MedicalNet is licensed under the MIT License, including its third-party component
3D-ResNets-PyTorch 3.0, Copyright (c) 2017 Kensho Hara.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.

Other dependencies (including MONAI and PyTorch) retain their own licenses. Upstream
software licensing is not permission for commercial exploitation of CT-RATE data.
