# CT-RATE 18 标签云端 Pilot 结果

## 结论

受控规模的 CT-RATE 多标签训练已在 Hugging Face Jobs 上完成。实验验证了唯读云端数据挂载、患者级划分、双窗 3D 预处理、MedicalNet 迁移学习、混合精度训练、校准和患者级 bootstrap 评估的端到端流程。该 pilot 样本极小，结果只能作为工程可行性证明，不能用于诊断或宣称临床有效性。

## 实验设置

- Job：[ct-rate-pilot-t4-16gb-medicalnet](https://huggingface.co/jobs/pitaya29/6a96b6e021c5aa7c8364b569)
- 运行环境：Hugging Face Jobs `t4-medium`，NVIDIA T4 16GB
- 成功任务运行时间：2,073 秒（约 34 分 33 秒）；调度时间 9 秒
- 模型：MedicalNet 预训练 3D ResNet-18，18 个 CT-RATE 标签
- 输入：肺窗和纵隔窗，目标尺寸 `96 × 192 × 192`，batch size 1，梯度累积 4
- 数据：训练 48 名患者/128 个 study；验证 7 名患者/16 个 study；测试 9 名患者/20 个 study
- 训练：最多 20 epoch，early-stopping patience 5；实际运行 10 epoch，最佳 epoch 为 5
- 隐私：CT 仅在 Hugging Face 以唯读 volume 挂载；未导出 manifest、影像路径、患者级预测或患者标识

计数说明：这里的 `study` 是适配器中的 volume 标识，包含同一扫描的不同重建，并非独立扫描数。
指标与 `split_summary.csv` 的各标签阳性数均按 volume 统计；bootstrap 才以患者为抽样单位。
测试集是官方验证集内划出的内部留出集，不是外部医院测试。实际模型卡见 [MODEL_CARDS.md](MODEL_CARDS.md)。

两次早期启动检查分别运行 90 秒和 47 秒后暴露并修复了只读源码挂载及 MedicalNet 下载依赖问题。三次 T4 总运行时间约 2,210 秒，按 US$0.60/小时估算约 US$0.37；排队和已取消的 L4 未计入 GPU 运行成本。

## 聚合指标

| 数据集 | 患者 | Study | Macro AUROC | Macro AUPRC | 灵敏度 | 特异度 | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 验证 | 7 | 16 | 0.6188 | 0.3390 | 0.9269 | 0.5138 | 0.3958 | 0.2503 |
| 测试 | 9 | 20 | 0.5198 | 0.2566 | 0.4345 | 0.4498 | 0.1420 | 0.2505 |

最佳验证 epoch 的选择分数为 0.6188。测试集中 Lung nodule 的 AUROC 为 0.9700（10 个阳性），但由极小验证集确定的阈值在测试集上灵敏度为 0，因此不能把高排序 AUROC 解读成可用的分类器。

## 结果文件

`results/ct_rate_pilot_medicalnet/` 包含：

- `history.csv`：10 个 epoch 的损失、验证 AUROC/AUPRC 和学习率
- `summary.json`：最佳 epoch、校准温度、阈值与验证/测试 macro 指标
- `split_summary.csv`：各划分的患者数、study 数及每个标签的阳性计数
- `val_metrics.json`、`test_metrics.json`：逐标签指标、混淆矩阵和患者级 bootstrap 区间
- ROC、PR 和 calibration 聚合图
- `resolved_config.yaml`：本次实验的完整可复现配置

126MB 的 `best.pt` 不进入 Git 历史。由于当前网络无法稳定完成单个 126MB 上传，[v0.3.0 Release](https://github.com/QihuiPan/ct-classification/releases/tag/v0.3.0) 将它无损拆成七个 `best.pt.partNN` 资产，并附带 `REASSEMBLE.md`、分片校验值和完整文件校验值；重组后的 SHA-256 为 `b99abc58677c8947d4ac6aaf4ed25ae4999a713b814c9e7c8edf765aefc2655d`。原始 CT、临时 manifest 和患者级预测不发布。

## 主要限制

- 测试集只有 9 名患者；置信区间很宽，无法支持泛化结论。
- 测试集中 Medical material、Pericardial effusion、Peribronchial thickening 和 Bronchiectasis 没有阳性样本，因此这些标签的 AUROC 未定义。
- 验证集仅 7 名患者，阈值与温度校准不稳定；温度达到配置上界 20.0。
- 这不是 21.3TB 全量训练。全量实验仍需要更大的云端存储、计算预算和独立外部验证。
- 数据集标签来自公开研究资源，不替代放射科医师复核；任何临床使用需另行完成伦理、隐私、法规和质量体系验证。
