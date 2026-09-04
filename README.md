# 3D CT 影像分类工程

这是一套面向研究和临床前验证的完整 CT 分类流水线，支持 DICOM 序列和 NIfTI 体数据。它覆盖：患者级数据划分、外部医院测试、3D 训练、单标签/多标签、类别不平衡处理、概率校准、患者级 bootstrap 置信区间、AUROC/AUPRC/灵敏度/特异度、混淆矩阵、3D Grad-CAM 和单病例推理。

> 该工程不能替代医生诊断。任何临床使用都需要独立的临床验证、隐私/伦理审查、质量管理和适用地区的合规评估。

## 交付状态（2026-09-04）

| 工作 | 状态 |
|---|---|
| COVID-CT-MD 全部 305 名患者：随机初始化基线与 MedicalNet fine-tune | 已训练、评估并发布 v0.2.0 |
| CT-RATE 16GB T4 pilot：48/7/9 名训练/验证/测试患者 | 已训练、评估并发布 v0.3.0 |
| 模型卡、聚合结果、checkpoint 校验和、发布边界 | 已提供，见下方链接 |
| CT-RATE 21.3TB 全量训练 | **未开始**，需要新的预算上限、云端作业权限和扩容验证 |
| 独立外部医院验证、临床验证 | **未完成**，没有可用于诊断的模型 |

阅读 [模型卡](MODEL_CARDS.md)、[CT-RATE 后续计划](CT_RATE_PLAN.md) 和
[第三方许可与隐私边界](THIRD_PARTY_NOTICES.md)。公开的是代码、汇总结果和研究模型；
原始 CT、患者清单、患者级预测、简历和 token 不公开。

## COVID-CT-MD 实验

本仓库包含 `configs/covid_ct_md.yaml` 和 `scripts/prepare_covid_ct_md.py`，用于公开的 [COVID-CT-MD](https://github.com/ShahinSHH/COVID-CT-MD) 三分类任务：Normal（76 人）、CAP（60 人）和 COVID-19（169 人）。配置把数据、预处理缓存、模型和训练结果放在 `E:/Codex/ct-classification`，这些大文件不会提交到 GitHub。来源、校验值和限制见 `DATASET_CARD_COVID_CT_MD.md`。

下载并解压官方 Figshare 文件后，生成患者级 manifest：

```powershell
python scripts/prepare_covid_ct_md.py `
  --dataset-root E:/Codex/ct-classification/datasets/COVID-CT-MD/raw `
  --output E:/Codex/ct-classification/datasets/COVID-CT-MD/manifest.csv
python scripts/validate_data.py --config configs/covid_ct_md.yaml
python scripts/train.py --config configs/covid_ct_md.yaml
```

也可以用 `scripts/setup_covid_ct_md.ps1` 续传官方压缩包、核对文件大小与 MD5、解压并生成 manifest。默认存储根目录就是 `E:/Codex/ct-classification`。

环境安装完成后，`scripts/run_covid_ct_md.ps1` 会把临时目录及 Matplotlib、PyTorch、Hugging Face 等缓存一并锁定到 E 盘，先验证全部 CT 再开始训练。

已完成的可复现实验、基线与 MedicalNet fine-tune 对比、患者级 bootstrap 置信区间及负面结果说明见 `RESULTS_COVID_CT_MD.md`；聚合图表和指标位于 `results/covid_ct_md/`。

脚本会核对已发表的 169/60/76 患者数，使用 DICOM SeriesInstanceUID 读取切片顺序，并从已脱敏的 DICOM 元数据保留性别和年龄组用于亚组评估。COVID-CT-MD 配置通过 MONAI 严格载入在 23 个医学数据集上预训练的官方 MedicalNet 3D ResNet-18 权重，将单通道首层权重平均扩展到肺窗和纵隔窗两个通道，再替换三分类头进行真正的 fine-tune。

## CT-RATE（21.3 TB）Pilot

仓库现已包含 CT-RATE 的 18 标签 pilot 配置、Hugging Face 存取/磁盘检查、受控大小的下载规划器和患者级 manifest 生成器。完整状态、官方标签、许可边界和硬件缺口见 `CT_RATE_PLAN.md`。

用户已选择**仅在 Hugging Face 云端处理 CT-RATE**，不要在本机执行下载或训练。
仓库保留的本机下载器仅用于其他明确选择本机路线的使用者；以下命令只做访问检查与下载规划：

```powershell
python scripts/check_ct_rate_access.py
python scripts/download_ct_rate_pilot.py
```

第二条命令默认只打印实际匹配文件数和大小；只有明确加 `--execute` 才下载。本项目不执行该选项。

当前采用的执行路线是 Hugging Face Jobs。`configs/ct_rate_hf_pilot.yaml` 和
`scripts/run_ct_rate_hf_cloud.py` 会把受控数据集作为唯读云端 volume 挂载，只按需读取
48 名训练患者和 16 名官方验证患者；CT 影像不会下载到本机。患者级 manifest、路径和预测
只存在于 Job 的临时磁盘，完成后仅同步模型、训练历史、聚合指标和图表。

16GB T4 云端 pilot 已完成：训练/验证/测试分别包含 48/7/9 名患者和 128/16/20 个
CT volume（包含同一扫描的不同重建），MedicalNet 3D ResNet-18 的最佳验证 macro AUROC 为 0.6188，内部留出测试 macro
AUROC 为 0.5198。完整实验记录、限制和复现信息见 `RESULTS_CT_RATE_PILOT.md`，聚合结果位于
`results/ct_rate_pilot_medicalnet/`。这是小样本工程验证，不是临床性能声明。

## 1. 数据准备

每一行代表一个 CT 检查。`image_path` 可以是一个 DICOM 序列目录，也可以是 `.nii` / `.nii.gz` 文件。

```csv
patient_id,study_id,image_path,site,label,split
P0001,S0001,D:/ct_data/P0001/S0001,Hospital_A,negative,train
P0002,S0002,D:/ct_data/P0002/S0002.nii.gz,Hospital_A,positive,val
P0003,S0003,D:/ct_data/P0003/S0003,Hospital_B,positive,test
```

- 先复制 `data/manifest_template.csv` 为 `data/manifest.csv`，再替换为真实路径和标签。
- 同一患者可以有多个检查，但必须具有同一个 `patient_id`。
- DICOM 必须先完成脱敏；不要把姓名、住院号等信息写入 manifest。
- 一个 `image_path` 目录最好只含一个 DICOM Series；如果同一目录有多套序列，请在 manifest 增加 `SeriesInstanceUID` 列，并设置 `data.dicom_series_id_column`，系统不会静默猜测使用哪一套序列。
- 单标签任务的 `label` 可以是配置中 `task.classes` 的名称或从 0 开始的类别编号。
- 多标签任务把 `task.type` 改成 `multi_label`，并为每个类别建立一个 0/1 列，例如 `nodule,effusion,emphysema`，同时修改 `label_columns`。

如果原始 manifest 没有 `split`，运行患者级划分：

```powershell
python scripts/prepare_manifest.py --config configs/ct_3d.yaml
```

`split.mode` 支持：

- `patient_random`：按患者随机且尽量分层地切分 train/val/test。
- `external_site`：`external_test_sites` 中的医院全部留作外部测试，其余患者再划分 train/val。
- `existing`：直接使用 manifest 已有的 `train`、`val`、`test`。

## 2. 安装

建议使用有 CUDA 的 NVIDIA GPU。先根据显卡环境从 PyTorch 官方安装选择器安装对应版本的 PyTorch，再安装本项目：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. 修改配置

打开 `configs/ct_3d.yaml`，至少修改：

- `task.type` 和 `task.classes`
- `data.manifest` 和 `data.label_columns`
- `data.windows`：必须根据目标病种确认窗宽窗位
- `data.target_spacing` 与 `data.target_size`
- `model.pretrained_checkpoint`：如有同结构的 3D 预训练权重，填入这里进行 fine-tune
- `evaluation.subgroup_columns`：默认按医院输出亚组结果，也可加入性别、年龄组、扫描仪型号等列

默认输入把肺窗、纵隔窗和骨窗堆叠为 3 个通道。不要未经医学确认就把这套窗设置用于所有病种。

默认关闭左右翻转增强；只有确认左右侧不会改变标签含义时才能设置非零的 `flip_probability`。

正式训练前先检查全部文件能否读取：

```powershell
python scripts/validate_data.py --config configs/ct_3d.yaml
```

再渲染一个预处理后的多窗质控图，人工检查方向、裁剪范围和窗设置：

```powershell
python scripts/preview_preprocessing.py `
  --config configs/ct_3d.yaml `
  --study-id S0001 `
  --output runs/ct_3d/preprocessing_qc.png
```

## 4. 训练

```powershell
python scripts/train.py --config configs/ct_3d.yaml
```

训练完成后，`runs/ct_3d` 中会包含：

- `best.pt`：模型、配置、温度校准参数和决策阈值
- `history.csv`：每个 epoch 的训练记录
- `val_predictions.csv`、`test_predictions.csv`
- `val_metrics.json`、`test_metrics.json`
- AUROC、AUPRC、校准曲线和混淆矩阵图

指标 JSON 中还包含 `subgroups`，分别报告不同医院及配置的其他亚组表现。

阈值只在 val 集上选择，然后原样应用于 test，避免测试集信息泄漏。

## 5. 独立评估

```powershell
python scripts/evaluate.py --checkpoint runs/ct_3d/best.pt --split test
```

评估会按患者进行 bootstrap，输出置信区间。若一个患者包含多次检查，抽样时这些检查会作为一个整体进入或离开样本。

## 6. 单病例推理与 Grad-CAM

```powershell
python scripts/infer.py `
  --checkpoint runs/ct_3d/best.pt `
  --image D:/ct_data/example_patient/series `
  --output runs/ct_3d/example_prediction.json `
  --gradcam runs/ct_3d/example_gradcam.png
```

如果目录中包含多套 DICOM 序列，再加上 `--series-id <SeriesInstanceUID>`。

输出包含每个类别的概率、最终判定、所用阈值和是否触发低置信度提示。Grad-CAM 是解释性辅助图，不是病灶分割结果。

## 7. 推荐的验证顺序

1. 先用 10–20 个已脱敏检查运行完整流程，确认方向、窗宽窗位和标签映射。
2. 人工抽查预处理后的关键切片，防止左右翻转、序列混用或错误裁剪。
3. 锁定内部测试集后再训练；测试集不可用于调参和选阈值。
4. 在不同医院、扫描仪或时间段进行外部测试。
5. 检查不同性别、年龄段、设备和采集协议上的亚组表现及概率校准。

已发布模型的实际情况见 [MODEL_CARDS.md](MODEL_CARDS.md)。`CLINICAL_VALIDATION_CHECKLIST.md`
和 `MODEL_CARD_TEMPLATE.md` 仍作为未来实验的清单与模板，不表示临床验证已完成。

## 8. 常见风险

- 同一患者的不同序列分到不同集合会造成严重的数据泄漏。
- 按切片训练并按切片随机划分，通常会高估真实性能。
- 只报告 Accuracy 会掩盖类别不平衡问题。
- Grad-CAM 看起来合理不代表模型判断正确。
- 如果标签来自报告文本，需要人工抽样核验标签噪声。
- CT 的重建核、层厚、增强期相和扫描设备差异可能造成域偏移。
