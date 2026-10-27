# 无卡前期准备记录

日期：2026-10-26。远端项目：`/root/onethingai-fs/projects/tango2-rdpo/`，位于 OneThingAI 持久盘 `/root/onethingai-fs/`。本地与 GitHub 仓库均为 `psh66666/tango2-rdpo`。

## 已完成

- 官方 Tango 源码固定在提交 `34ecd38898508017ca937508a047b42817d2b070`，保持原样放在 `upstream/tango/`。
- Audio-Alpaca 与 Tango 起始模型固定到 [`sources.json`](sources.json) 的 revision；29 个下载文件的实际 SHA-256 与大小见 [`metadata/snapshot_hashes.json`](metadata/snapshot_hashes.json)。数据、权重和临时下载地址均未提交 Git。
- 现有 Python 3.11 无卡环境的主要包版本见 [`environment.lock`](environment.lock)。PyTorch 为 `2.4.0+cu121`；无卡模式下 `torch.cuda.is_available()` 为 `False`。
- 15,025 条 Audio-Alpaca 数据全部转为 16 kHz 单声道 WAV，拒绝 0 条，音频留在远端 `data/processed/audio/`。
- 根据同一或相近提示词、重复音频分组后，训练/验证/测试分别为 12,019 / 1,504 / 1,502 条；分组时合并了 368 对近似提示词。冻结切分的 ID 与哈希见 [`metadata/splits.csv`](metadata/splits.csv)。
- 自动审计发现跨切分相同提示词 0、相同音频 0、已识别的近似提示词交叉 0；40 条外部挑战提示与数据集的近重复 0。审计详情见 [`metadata/audit.json`](metadata/audit.json)。相似度规则是启发式检查，不能保证识别所有语义改写。
- 远端 `outputs/data_review/` 已准备 30 对训练偏好音频和待填写的 `review_template.csv`。标签合理性仍需人耳确认。
- 本组脚本已做 Python 语法检查。按照用户安排，模型前向、训练、生成和听评测试留到开卡后由用户执行；目前没有 GPU 实测结果。

## 开卡后的环境检查（2026-10-27）

- RTX 4090 24 GiB 已识别，当前 PyTorch 2.4.0+cu121 能访问 CUDA 且支持 bf16；检查时无其他 GPU 计算进程。
- `/root/onethingai-fs/` 仍是 JuiceFS 挂载点，剩余约 190 GiB；远端项目代码为提交 `94b2e57`，与关机前一致。
- Python 语法、YAML 配置路径、调度器的 1000 步、`scaled_linear`、`v_prediction` 字段通过静态检查；切分审计仍显示零交叉泄漏。
- 尚未加载模型、运行前向、训练或生成；没有显存及吞吐实测值。

## 下一步：GPU 探针

当前显卡可用于 **SFT 与 DPO 的 2 步探针**，再做各 100 步探针。24 GiB 显存尚不能保证装得下模型及冻结参考 UNet；先测峰值显存和更新耗时，再决定是否使用 48 GiB 卡或统一缩小训练预算。正式 B/I/A 训练只在探针和固定验证集诊断通过后开始。精确命令见 [`README.md`](README.md)。

保留大于 100 GiB 的持久盘空余，并确保代码、模型、训练输出都在上述持久盘路径。当前模型与数据快照及预处理音频均已在持久盘，无需重新下载。
