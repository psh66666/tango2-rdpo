# Tango 2 偏好优化课程实验

本仓库研究一个可证伪的问题：Tango 2 的相对 DPO 目标可能让劣选音频的去噪误差低于冻结参考模型。我们在原 DPO 上加一个只对该现象起作用的平方铰链约束，比较共同 SFT 起点后的原版 DPO、改进版和约束强度消融。误差是拟合代理量，不是精确生成概率。正式效果要靠保留提示词的匿名盲听判断。

**状态：共同 SFT、B/I/A 三组正式训练、固定测试集机制诊断、统一音频生成和匿名盲听材料均已完成；等待真实听评。** 正式预算与配置见 [`experiment_registry.json`](experiment_registry.json)，测试协议见 [`evaluation_protocol.json`](evaluation_protocol.json)。机制指标不能代替生成音质或真实听评。

## 来源和许可

主论文：[Tango 2](https://doi.org/10.1145/3664647.3681688)；方法启发：[RDPO](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Rethinking_DPO-style_Diffusion_Aligning_Frameworks_ICCV_2025_paper.html)。精确源码提交、数据与模型 revision 见 [`sources.json`](sources.json)。官方 Tango 源码固定为未修改的独立依赖，不提交本仓库；其许可是 CC BY-NC-ND 4.0。起始模型为 CC BY-NC-SA 4.0，Audio-Alpaca 数据为 Apache-2.0。权重、原始音频和训练输出均不推送 GitHub。

## 部署布局

远端持久盘根目录：`/root/onethingai-fs/projects/tango2-rdpo`。脚本、`models/`、`data/`、`runs/`、`outputs/` 均放这里；T5 缓存在同一持久盘的 `/root/onethingai-fs/cache/huggingface/hub`。`upstream/tango` 固定到 `34ecd38898508017ca937508a047b42817d2b070`。现有 Python 3.11 环境在 `/root/miniconda3/envs/loopscape-py311`，实际软件版本见 [`environment.lock`](environment.lock)。环境本身可能随实例重建，持久盘中的数据与代码不会依赖实例临时目录。

GPU 探针使用独立环境 `/root/tango2-rdpo-env`，继承现有 CUDA/PyTorch 环境，只覆盖与固定 Tango 源码不兼容的依赖；精确版本见 [`environment.lock`](environment.lock)。实例重建时执行：

```bash
/root/miniconda3/envs/loopscape-py311/bin/python -m venv --system-site-packages /root/tango2-rdpo-env
/root/tango2-rdpo-env/bin/python -m pip install --no-deps diffusers==0.18.2 huggingface-hub==0.25.2 librosa==0.9.2
```

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
/root/miniconda3/envs/loopscape-py311/bin/python scripts/export_snapshot_hashes.py \
  --dataset-manifest data/snapshots/audio-alpaca/snapshot_manifest.json \
  --model-manifest models/tango-base/snapshot_manifest.json \
  --output metadata/snapshot_hashes.json
/root/miniconda3/envs/loopscape-py311/bin/python scripts/make_label_review.py \
  --train-manifest data/processed/manifests/train.jsonl \
  --output-dir outputs/data_review
/root/miniconda3/envs/loopscape-py311/bin/python -m compileall -q tango2_rdpo scripts
```

数据提取后将 16 kHz 单声道 WAV 留在 `data/processed/audio/`，输出按相同或相近提示词及重复音频分组的 80/10/10 清单、拒绝记录和审计。`data/` 整体默认忽略，不把含本机绝对路径的训练清单上传到 GitHub。组间泄漏审计结果必须为零后才允许探针。

## 开卡后的实验顺序

2026-10-27 已在 RTX 4090 24 GiB 上完成 SFT/DPO 的 2 步及 100 步探针；SFT 与 DPO 峰值显存分别约 2.74/4.55 GiB。100 步有效训练时间约 120/127 秒。基座模型已生成两段 10.242 秒 WAV，均可解码且非静音。以下全部在远端项目根目录，并使用锁定的 Python 环境运行：

```bash
cd /root/onethingai-fs/projects/tango2-rdpo
export PATH=/root/tango2-rdpo-env/bin:$PATH
python -m tango2_rdpo.train --config configs/sft.yaml --max-updates 2 --output-dir runs/probe_sft_2
python -m tango2_rdpo.train --config configs/dpo_baseline.yaml --init runs/probe_sft_2/final --max-updates 2 --output-dir runs/probe_dpo_2
python -m tango2_rdpo.train --config configs/sft.yaml --max-updates 100 --output-dir runs/probe_sft_100
python -m tango2_rdpo.train --config configs/dpo_baseline.yaml --init runs/probe_sft_100/final --max-updates 100 --output-dir runs/probe_dpo_100
python -m tango2_rdpo.train --config configs/sft.yaml --output-dir runs/sft
python -m tango2_rdpo.train --config configs/dpo_baseline.yaml --init runs/sft/final --output-dir runs/baseline
python -m tango2_rdpo.train --config configs/dpo_regularized.yaml --init runs/sft/final --output-dir runs/regularized
python -m tango2_rdpo.train --config configs/dpo_ablation.yaml --init runs/sft/final --output-dir runs/ablation
```

正式训练采用已锁定的 5000 对：SFT 学习率 `1e-5`、1 轮；B/I/A 的 DPO 学习率均为 `5e-5`、2 轮，`lambda_reject` 分别为 0、0.1、0.3。其余 DPO 字段相同，均从同一 `runs/sft/final` adapter 出发。选参只用了固定验证子集，未使用测试集；选择依据和较好音频代理量的取舍见 [实验注册表](experiment_registry.json)。最终效果仍需匿名盲听。

正式训练均按预算完成：SFT 1250 步，B/I/A 各 2500 步；同一 SFT 起点与训练源码提交 `7788161` 经运行状态核对。四组日志内训练时间合计 11634.07 秒（3.23 GPU 小时），按 2 元/小时约 6.46 元；模型加载、诊断、生成和探针另计。固定测试集 200 个偏好对、每对三组噪声上，I 相对 B 的劣选拟合代理风险由 22.67% 降至 5.00%，但较好音频的去噪误差增加 0.00212；[完整配对统计与区间](reports/metrics/dpo_test200_comparison.json)已保留。这个取舍是否改善实际音频，仍待匿名听评。

固定验证或测试偏好对的机制检查：

```bash
python -m tango2_rdpo.diagnose --config configs/dpo_baseline.yaml \
  --manifest data/processed/manifests/val.jsonl --adapter runs/baseline/final \
  --output outputs/metrics/baseline_val.json
python -m tango2_rdpo.diagnose --config configs/dpo_regularized.yaml \
  --manifest data/processed/manifests/val.jsonl --adapter runs/regularized/final \
  --output outputs/metrics/regularized_val.json
```

按锁定协议完成测试集 200 个提示 × P/S/B/I/A 五组模型，共 1000 条 WAV；挑战集 40 个提示 × B/I 两组，共 80 条 WAV。所有文件均为 10.242 秒；[测试集审计](reports/metrics/test_wav_audit.json)和[挑战集审计](reports/metrics/challenge_wav_audit.json)都记录了 0 条静音、0 条达到 0.999 峰值阈值的近削波文件。100 组测试集匿名配对与 40 组挑战集匿名配对已复制到本地 `outputs/listening_test/participant/` 和 `outputs/listening_challenge/participant/`；音频及私有答案键不上传 GitHub。听评说明见[听评指南](reports/listening_instructions.md)。后处理状态见[运行记录](reports/metrics/posttrain_status.json)。

选定最终检查点后，使用统一提示、种子 1200 和 200 步生成，复现匿名盲听表的命令为：

```bash
python -m tango2_rdpo.generate --config configs/eval.yaml \
  --prompt-manifest data/processed/manifests/eval_prompts.jsonl --output-dir outputs/test_wav
python -m tango2_rdpo.blind_listening \
  --generation-manifest outputs/test_wav/generation_manifest.jsonl \
  --output-dir outputs/listening_test --count 100
python -m tango2_rdpo.evaluate --ratings outputs/listening_test/ratings_filled.csv \
  --answer-key outputs/listening_test/private/answer_key.json \
  --output outputs/metrics/blind_listening.json
```

盲听表 `participant/ratings_template.csv` 的 `alignment_choice` 和 `naturalness_choice` 分别填 `L`、`R` 或 `T`；原始填写表须保留，`private/answer_key.json` 不分发评审者。听评者至少 3 人、100 对固定 B/I 配对；按提示词 bootstrap 给出 95% 区间。若区间包含 0.5，只能报告证据不足。

完整实验设计见上层课程项目的执行计划书。[GPU 时间与费用记录](reports/metrics/gpu_cost_estimate.json)：正式训练日志 3.23 小时约 6.46 元，后处理实际运行窗口 2.94 小时约 5.89 元，两项合计约 12.35 元；实例从系统启动到后处理完成约 8.04 小时，对应约 16.07 元。系统启动时间不是平台计费起点，实例仍在运行时费用继续增加，实际账单须以平台记录为准。训练脚本会在 CUDA 不可用时拒绝运行，避免无卡模式误开始训练。
