# 实验准备与 GPU 探针记录

日期：2026-10-26。远端项目：`/root/onethingai-fs/projects/tango2-rdpo/`，位于 OneThingAI 持久盘 `/root/onethingai-fs/`。本地与 GitHub 仓库均为 `psh66666/tango2-rdpo`。

## 已完成

- 官方 Tango 源码固定在提交 `34ecd38898508017ca937508a047b42817d2b070`，保持原样放在 `upstream/tango/`。
- Audio-Alpaca 与 Tango 起始模型固定到 [`sources.json`](sources.json) 的 revision；29 个下载文件的实际 SHA-256 与大小见 [`metadata/snapshot_hashes.json`](metadata/snapshot_hashes.json)。数据、权重和临时下载地址均未提交 Git。
- 现有 Python 3.11 无卡环境的主要包版本见 [`environment.lock`](environment.lock)。PyTorch 为 `2.4.0+cu121`；无卡模式下 `torch.cuda.is_available()` 为 `False`。
- 15,025 条 Audio-Alpaca 数据全部转为 16 kHz 单声道 WAV，拒绝 0 条，音频留在远端 `data/processed/audio/`。
- 根据同一或相近提示词、重复音频分组后，训练/验证/测试分别为 12,019 / 1,504 / 1,502 条；分组时合并了 368 对近似提示词。冻结切分的 ID 与哈希见 [`metadata/splits.csv`](metadata/splits.csv)。
- 自动审计发现跨切分相同提示词 0、相同音频 0、已识别的近似提示词交叉 0；40 条外部挑战提示与数据集的近重复 0。审计详情见 [`metadata/audit.json`](metadata/audit.json)。相似度规则是启发式检查，不能保证识别所有语义改写。
- 远端 `outputs/data_review/` 已准备 30 对训练偏好音频和待填写的 `review_template.csv`。标签合理性仍需人耳确认。
- 本组脚本完成 Python 语法检查；开卡后已运行小规模训练和生成探针，正式训练与人耳听评另见下文。

## 开卡后的环境检查（2026-10-27）

- RTX 4090 24 GiB 已识别，当前 PyTorch 2.4.0+cu121 能访问 CUDA 且支持 bf16；检查时无其他 GPU 计算进程。
- `/root/onethingai-fs/` 仍是 JuiceFS 挂载点，剩余约 190 GiB；远端项目代码为提交 `94b2e57`，与关机前一致。
- Python 语法、YAML 配置路径、调度器的 1000 步、`scaled_linear`、`v_prediction` 字段通过静态检查；切分审计仍显示零交叉泄漏。
- 当时的检查仅确认环境；下节记录随后完成的模型前向、训练和生成结果。

## GPU 探针启动记录

- 首次 SFT 启动在导入阶段遇到固定 Tango 源码与 Diffusers 0.30.0 的接口不兼容；独立环境改用官方要求的 Diffusers 0.18.2，配套 Hugging Face Hub 0.25.2。
- 后续加载发现固定源码用位置参数调用 Librosa 的 `pad_center`，但官方要求的 Librosa 0.9.2 已要求关键字参数；本组加载器只在检测到该接口时适配调用，官方源码保持原样。
- 官方 VAE 用 float32 噪声采样潜变量；进入 bf16 UNet 前已统一类型。官方声码器导出 NumPy 前需要 float32；生成阶段单独使用 float32 解码。
- SFT 和 DPO 的 2 步探针均保存了最终 adapter 与优化器状态；SFT 已验证从第 1 步恢复到第 2 步。3 项偏好损失公式检查通过。
- 100 步 SFT 用时 119.68 秒、峰值显存 2.738 GiB；100 步 DPO 候选用时约 127–131 秒、峰值显存 4.552 GiB。基座生成两段 10.242 秒、16 kHz 单声道且非静音的 WAV。
- 用相同 64 对验证样本、每对 3 组固定噪声选参：DPO 学习率 `5e-5` 优于 `1e-5` 的偏好代理指标；在 `5e-5` 下，`λ=0/0.1/0.3` 的劣选拟合风险比例为 0.516/0.172/0.094。非零约束同时使较好音频代理误差略升；选择较弱的 0.1 作为主改进，0.3 作消融，真实音质仍待听评。

## 下一步：正式训练

推荐档预算已在 [`experiment_registry.json`](experiment_registry.json) 冻结：5000 对、共同 SFT 1 轮、B/I/A 各 2 轮。预计纯训练约 3.09 GPU 小时，不含模型加载、验证、生成和听评；尚缺租机单价，不能换算费用。精确命令见 [`README.md`](README.md)。

保留大于 100 GiB 的持久盘空余，并确保代码、模型、训练输出都在上述持久盘路径。当前模型与数据快照及预处理音频均已在持久盘，无需重新下载。
