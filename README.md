# Tango 2 偏好优化课程实验

本仓库研究一个可证伪的问题：Tango 2 的相对 DPO 目标可能让劣选音频的去噪误差低于冻结参考模型。我们在原 DPO 上加一个只对该现象起作用的平方铰链约束，比较共同 SFT 起点后的原版 DPO、改进版和约束强度消融。误差是拟合代理量，不是精确生成概率。正式效果要靠保留提示词的匿名盲听判断。

**状态：前期准备。** 数据、模型、脚本和环境正在锁定；尚未使用 GPU，所有超参数都是探针起点，没有训练结果。

## 来源和许可

主论文：[Tango 2](https://doi.org/10.1145/3664647.3681688)；方法启发：[RDPO](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Rethinking_DPO-style_Diffusion_Aligning_Frameworks_ICCV_2025_paper.html)。精确源码提交、数据与模型 revision 见 [`sources.json`](sources.json)。官方 Tango 源码固定为未修改的独立依赖，不提交本仓库；其许可是 CC BY-NC-ND 4.0。起始模型为 CC BY-NC-SA 4.0，Audio-Alpaca 数据为 Apache-2.0。权重、原始音频和训练输出均不推送 GitHub。

## 部署布局

远端持久盘根目录：`/root/onethingai-fs/projects/tango2-rdpo`。脚本、`models/`、`data/`、`runs/`、`outputs/` 均放这里；T5 缓存在同一持久盘的 `/root/onethingai-fs/cache/huggingface/hub`。`upstream/tango` 固定到 `34ecd38898508017ca937508a047b42817d2b070`。现有 Python 3.11 环境在 `/root/miniconda3/envs/loopscape-py311`，实际软件版本见 [`environment.lock`](environment.lock)。环境本身可能随实例重建，持久盘中的数据与代码不会依赖实例临时目录。

服务器无法直接连 Hugging Face 页面时，在可访问的本机生成短时效文件地址，再将 `data/*download-plan.json` 传到远端同名目录；远端可直连文件 CDN。下载器对 LFS 文件按来源 SHA-256 核验并支持断点续传。下载计划含临时签名地址，文件权限为 `0600`，不要提交或公开：

```bash
python scripts/download_snapshot.py audio-alpaca --output-root data/snapshots \
  --write-plan data/audio-alpaca-download-plan.json
python scripts/download_snapshot.py tango-base --output-root models \
  --write-plan data/tango-base-download-plan.json
# 将两份计划及小型 README/JSON 配置文件复制到远端持久盘后：
python scripts/download_snapshot.py audio-alpaca --output-root data/snapshots \
  --plan data/audio-alpaca-download-plan.json --workers 4
python scripts/download_snapshot.py tango-base --output-root models \
  --plan data/tango-base-download-plan.json --workers 2
```

官方配置指向 `stabilityai/stable-diffusion-2-1` 调度器；该路径目前无法匿名获取。本项目已将公开的 `sd2-community/stable-diffusion-2-1` 同类配置保存到 `models/scheduler/scheduler/scheduler_config.json`，使用前要在 GPU 探针时核对 1000 步、`scaled_linear`、`v_prediction` 等关键字段。这个来源替换须在报告中说明。

## 无卡阶段

在远端项目根目录运行；这些命令是数据准备和记录工作，不开始训练：

```bash
/root/miniconda3/envs/loopscape-py311/bin/python -m tango2_rdpo.prepare_data \
  --snapshot data/snapshots/audio-alpaca --output-dir data/processed
/root/miniconda3/envs/loopscape-py311/bin/python -m tango2_rdpo.audit_data \
  --manifest-dir data/processed/manifests \
  --challenge-manifest evaluation/challenge_prompts.jsonl
/root/miniconda3/envs/loopscape-py311/bin/python scripts/export_split_metadata.py \
  --manifest-dir data/processed/manifests --output-dir metadata
/root/miniconda3/envs/loopscape-py311/bin/python -m compileall -q tango2_rdpo scripts
```

数据提取后将 16 kHz 单声道 WAV 留在 `data/processed/audio/`，输出按相同或相近提示词及重复音频分组的 80/10/10 清单、拒绝记录和审计。`data/` 整体默认忽略，不把含本机绝对路径的训练清单上传到 GitHub。组间泄漏审计结果必须为零后才允许探针。

## 开卡后的实验顺序

先在 GPU 实例确认 CUDA 可用、持久盘挂载、模型能加载。一个 24 GiB 级单卡是**待实测候选**，目前没有显存或速度保证；先跑 2 步再跑 100 步探针，确定 bf16、LoRA 和训练预算。以下全部在远端项目根目录运行：

```bash
python -m tango2_rdpo.train --config configs/sft.yaml --max-updates 2 --output-dir runs/probe_sft
python -m tango2_rdpo.train --config configs/sft.yaml --output-dir runs/sft
python -m tango2_rdpo.train --config configs/dpo_baseline.yaml --init runs/sft/final --max-updates 2 --output-dir runs/probe_dpo
python -m tango2_rdpo.train --config configs/dpo_baseline.yaml --init runs/sft/final --output-dir runs/baseline
python -m tango2_rdpo.train --config configs/dpo_regularized.yaml --init runs/sft/final --output-dir runs/regularized
python -m tango2_rdpo.train --config configs/dpo_ablation.yaml --init runs/sft/final --output-dir runs/ablation
```

正式训练前必须先用共同的训练集小试和验证集诊断锁定学习率、`λ`、更新数，并在 `experiment_registry.json` 留档。B/I/A 从同一 `runs/sft/final` adapter 出发；`lambda_reject` 分别为 0、候选值和另一非零值；其余训练字段保持一致。配置内当前的 `5000` 条、`β=2000`、rank 8、学习率 `1e-5`、`λ=0.3/1.0` 均未经过 GPU 验证。

固定验证或测试偏好对的机制检查：

```bash
python -m tango2_rdpo.diagnose --config configs/dpo_baseline.yaml \
  --manifest data/processed/manifests/val.jsonl --adapter runs/baseline/final \
  --output outputs/metrics/baseline_val.json
python -m tango2_rdpo.diagnose --config configs/dpo_regularized.yaml \
  --manifest data/processed/manifests/val.jsonl --adapter runs/regularized/final \
  --output outputs/metrics/regularized_val.json
```

选定最终检查点后，统一提示、种子和 200 步生成，准备匿名盲听表：

```bash
python -m tango2_rdpo.generate --config configs/eval.yaml \
  --prompt-manifest data/processed/manifests/eval_prompts.jsonl --output-dir outputs/wav
python -m tango2_rdpo.blind_listening \
  --generation-manifest outputs/wav/generation_manifest.jsonl --output-dir outputs/listening
python -m tango2_rdpo.evaluate --ratings outputs/listening/ratings_filled.csv \
  --answer-key outputs/listening/private/answer_key.json \
  --output outputs/metrics/blind_listening.json
```

盲听表 `participant/ratings_template.csv` 的 `alignment_choice` 和 `naturalness_choice` 分别填 `L`、`R` 或 `T`；原始填写表须保留，`private/answer_key.json` 不分发评审者。听评者至少 3 人、100 对固定 B/I 配对；按提示词 bootstrap 给出 95% 区间。若区间包含 0.5，只能报告证据不足。

完整实验设计见上层课程项目的执行计划书。计算成本要用 100 步探针实测吞吐后估算。训练脚本会在 CUDA 不可用时拒绝运行，避免无卡模式误开始训练。
