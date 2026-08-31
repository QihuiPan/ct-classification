# CT-RATE 训练准备与当前边界

状态日期：2026-08-31

## 官方任务

CT-RATE 官方数据集包含 21,304 名患者的 25,692 次非增强胸部 CT 扫描，经不同重建扩展为 50,188 个 NIfTI 体数据，总大小约 21.3 TB。官方患者划分为 20,000 名训练患者和 1,304 名验证患者。数据还包含放射学报告、元数据和 18 个报告级多异常二进制标签。

稳定标签顺序来自官方 CT-CLIP 实现：

1. Medical material
2. Arterial wall calcification
3. Cardiomegaly
4. Pericardial effusion
5. Coronary artery wall calcification
6. Hiatal hernia
7. Lymphadenopathy
8. Emphysema
9. Atelectasis
10. Lung nodule
11. Lung opacity
12. Pulmonary fibrotic sequela
13. Pleural effusion
14. Mosaic attenuation pattern
15. Peribronchial thickening
16. Consolidation
17. Bronchiectasis
18. Interlobular septal thickening

官方来源：

- 数据集：https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
- 官方实现：https://github.com/ibrahimethemhamamci/CT-CLIP
- 论文：https://arxiv.org/abs/2403.17834

数据集采用 CC-BY-NC-SA-4.0，只用于符合条款的非商业研究；原始体数据、报告、元数据和标签不得提交到本 GitHub 仓库。

## 本机核对结果

- E 盘可用空间约 1.02 TB。
- 完整数据本身为 21.3 TB；按 15% 安全余量计算，需要约 24.5 TB 可用空间，尚未包含预处理缓存、检查点和中间结果。
- 本机 Hugging Face CLI 当前未登录，也尚未证明此账号已接受 CT-RATE 的受控访问条款。
- RTX 4080 SUPER 具有 16 GB 显存，适合先运行降低空间分辨率的 MedicalNet 3D ResNet-18 pilot；不适合在本机单卡完整复现 CT-CLIP 的 480x480x240 训练。

因此，当前不能诚实地声称“21.3 TB 已下载并完成训练”。直接开始完整下载会在约 5% 进度前耗尽 E 盘。

## 已完成的工程准备

- `scripts/check_ct_rate_access.py`：检查 Hugging Face 登录/仓库可见性和完整下载磁盘缺口。
- `scripts/download_ct_rate_pilot.py`：先计算准确下载文件数与大小，确认 20% 空间余量后才允许执行有限患者 pilot。
- `scripts/prepare_ct_rate.py`：读取官方 NIfTI 路径和 18 标签 CSV，按患者生成无泄漏 manifest。
- `configs/ct_rate_pilot.yaml`：面向 16 GB GPU 的 18 标签 MedicalNet pilot 配置，所有缓存和训练结果位于 E 盘。
- `tests/test_ct_rate.py`：锁定标签顺序、文件名解析和患者隔离行为。

## 获得授权后的执行顺序

先在 CT-RATE 页面接受数据条款，并使用具有读取权限的 Hugging Face token 登录：

```powershell
E:/Codex/ct-classification/venv/Scripts/hf.exe auth login
python scripts/check_ct_rate_access.py
```

完整 21.3 TB 下载只有在提供至少约 25 TB 可用存储后才开始。当前 E 盘可先规划一个有限 pilot：

```powershell
python scripts/download_ct_rate_pilot.py
python scripts/download_ct_rate_pilot.py --execute
python scripts/prepare_ct_rate.py
python scripts/validate_data.py --config configs/ct_rate_pilot.yaml
python scripts/train.py --config configs/ct_rate_pilot.yaml
```

下载脚本默认只选择 48 名训练患者和 16 名官方验证患者；先显示实际匹配文件数、NIfTI 数量和字节数，未加 `--execute` 时不会下载。

`prepare_ct_rate.py` 保留官方训练患者作为 train，并按稳定患者哈希把官方验证患者分成 local val/test；阈值只在 local val 上选择，local test 不参与调参。

## 完整训练所需外部资源

要继续完整数据训练，需要：

1. 用户本人接受 Hugging Face 受控访问与数据许可条款。
2. 至少约 25 TB 可用对象存储或本地 NVMe/HDD；若保存预处理缓存，建议 30 TB 以上。
3. 多 GPU 或云端分布式训练环境；单张 16 GB GPU 只作为 pilot 和代码验证设备。
4. 明确的预算与云账户授权。购买或创建付费资源不能由本项目自动推断执行。

在这些条件满足前，本仓库只提交代码、配置、测试和非患者级汇总结果，不上传 CT-RATE 原始数据或报告。
