# CT-RATE 训练准备与当前边界

状态日期：2026-09-04。**16GB T4 pilot 已完成；21.3TB 全量训练未开始。**

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

## 当前执行范围

用户已接受受控访问条款，并选择 CT-RATE 只在 Hugging Face 云端运行。
不在本机下载、缓存或训练 CT-RATE。COVID-CT-MD 的既有本机数据和训练环境仍在 E 盘。

已完成的 pilot 使用 48 名训练患者、7 名验证患者和 9 名内部测试患者，分别对应
128、16 和 20 个重建体数据。`study_id` 在此数据适配器里实际标识一个 volume，
不是独立患者，也不一定是独立扫描。完整报告见 [RESULTS_CT_RATE_PILOT.md](RESULTS_CT_RATE_PILOT.md)。

## 已完成的工程准备

- `scripts/check_ct_rate_access.py`：检查 Hugging Face 登录/仓库可见性和完整下载磁盘缺口。
- `scripts/download_ct_rate_pilot.py`：先计算准确下载文件数与大小，确认 20% 空间余量后才允许执行有限患者 pilot。
- `scripts/prepare_ct_rate.py`：读取官方 NIfTI 路径和 18 标签 CSV，按患者生成无泄漏 manifest。
- `configs/ct_rate_pilot.yaml`：面向 16 GB GPU 的 18 标签 MedicalNet pilot 配置，所有缓存和训练结果位于 E 盘。
- `configs/ct_rate_hf_pilot.yaml`：已在 Hugging Face Jobs T4 16 GB GPU 验证的云端 pilot 配置。
- `scripts/run_ct_rate_hf_cloud.py`：从唯读 CT-RATE 云端 volume 构建有限 manifest 并训练，不导出患者级工件。
- `tests/test_ct_rate.py`：锁定标签顺序、文件名解析和患者隔离行为。

## Hugging Face Jobs 路线

当前路线不会把 CT-RATE 影像下载到本机。Job 将官方数据集挂载为唯读 volume，文件仅在云端按需读取；
源代码和一个空输出目录经私有 `jobs-artifacts` bucket 挂载。Job 的 manifest、患者路径、患者级预测
和预处理缓存都写在临时磁盘，完成后只拉回 checkpoint、聚合指标、图表和训练历史。

已完成作业使用 `t4-medium`（16GB T4、100GB 临时盘），成功运行 2,073 秒。
L4 路线只曾排队，后被取消；不要把旧 L4 参数当作本次实验硬件。
启动脚本固定默认 48 名训练患者和 16 名官方验证患者，不会自动扩展为全量训练。

运行容器为 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`，源码只读挂载到 `/workspace`，
数据只读挂载到 `/mnt/ct-rate`，私有输出挂载到 `/outputs`。容器中执行
`bash /workspace/scripts/run_ct_rate_hf_cloud.sh`。启动脚本的依赖版本范围并非完整锁文件，
不能保证未来解析出逐字节相同的环境；后续训练须保存容器 digest 和实际依赖清单。

临时写入/作业 token 已在 pilot 完成后撤销。重新启动需要新的最小权限凭证，
通过 Hugging Face 安全登录或 secret 管理注入，不能写入代码、命令日志或聊天。

`prepare_ct_rate.py` 保留官方训练患者作为 train，并按稳定患者哈希把官方验证患者分成
内部 val/test；阈值只在 val 上选择，test 不参与调参。这不构成外部医院测试。

## 完整训练所需外部资源

要继续完整数据训练，需要：

1. 明确新的**总费用上限**（GPU、存储及适用的传输费用）和单次作业超时。
   pilot 的费用不是全量报价；不能把以前的小额授权当成无限预算。
2. 恢复最小权限的云端作业/私有输出写入凭证，确认账号余额和受控数据访问仍有效。
3. 验证云端按需挂载的吞吐量、缓存与磁盘上限。按需读取不必先复制全部 21.3TB；
   只有选择完整独立副本时才需要约 25TB 或以上原始数据空间。
4. 改造并验证缓存/断点恢复。当前预处理磁盘缓存无淘汰策略，不能将 pilot 患者数直接改成
   20,000/1,304 后在 100GB 临时盘上运行。两个 float32 窗、96×192×192 的 50,188 个
   张量在压缩前合计约 1.42TB；真实压缩率需实测。`cache_dir: null` 可以关闭项目缓存，
   但会重复预处理，且不能解决云挂载客户端自身的缓存上限。
5. 在获批预算内先测更大子集的吞吐、磁盘峰值和显存，再估计全量训练时间。
   16GB 卡可以处理当前分辨率，但没有证据证明单卡全量训练的时间/费用可接受。
   仓库当前不实现分布式训练；也不声称复现 CT-CLIP。
6. 锁定划分与评估协议，保存数据 revision、代码 SHA、环境和实际各类患者/volume 数；
   处理无阳性标签、重建重复权重和小样本阈值失稳。保持内部测试集不参与调参，
   另外申请独立外部验证数据。

以上未完成项是下一阶段的准入条件，不是已经完成的训练。本轮不会自动提交付费作业。
原始 CT-RATE 数据和报告始终不上传 GitHub。
