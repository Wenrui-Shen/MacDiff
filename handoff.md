# MacDiff 交接：全局/六部位文本、CLIP 缓存与 Stage1 训练（2026-10-01）

本文面向完全没有上下文的新会话。**当前主任务是把旧的逐人物描述/BPE local 文本，升级为逐样本、逐人物的简短 global + 六个固定部位描述，并接入可复用的 CLIP 句级特征缓存和 Stage1 训练。** 用户在后续消息中已确认最新版文本特征 cache 生成完成；不要重新生成。下一步是同步下述续训快照修复、保留完整缓存校验做真实双卡 smoke，再训练和比较 LP。

这是本会话结束时的当前状态。旧交接完整保存在 `handoff_artifacts/handoff_before_20261001_refresh.md`，仅供历史追溯；其中“不要重新生成文本/缓存”“还需手动同步性能版文件”“当前 variance 是保存 global 目标”等说法已经被本文件更新。详细链路检查见 `tools/vlm_pilot/TEXT_CACHE_CHAIN_AUDIT.md`。

## 1. 当前做到哪里、还缺什么

| 项目 | 当前状态 |
|---|---|
| 简短英文提示词及中文翻译 | 已完成；固定 global + 六部位，不再以 tokenizer 的词元当作 local 描述 |
| 文本输出清理 | 已完成；每个样本一行，只含序号和描述；模型/设备/运行设置单独保存 |
| 双卡生成、按序号合并、自动 resume | 代码及 CPU 模拟验证已完成；尚无本会话内的真实双卡全量完成证据 |
| 旧文本→缓存→训练链检查 | 已完成代码检查；人物配对等历史修正已经确认 |
| 新版 v3 CLIP 缓存生成/恢复/校验 | 用户已确认生成完成；本地没有服务器文件，首次训练保留完整启动校验 |
| v3 reader、训练保护和两份新配置 | 已接入；尚无真实 GPU/DDP smoke 或新版 LP 结果 |
| 训练日志调整 | 已完成；保留 uni，删除 empty 骨架日志及 target drift MSE，改 global+local 方差 |
| 服务器 Git 同步 | 用户已执行成功，HEAD 为 `edc924a`；网络问题已解除 |

**后续完整链路检查没有发现首次训练的阻塞错误；发现并修复了续训快照残留问题。** 目标库先于模型 checkpoint 保存，中断可能留下同名目标库或临时文件；现在仅在模型 checkpoint 不存在时原子重建孤立快照，已配对的 checkpoint 不允许覆盖。修复位于 `util/sample_text_target_bank.py` 和 `util/shared_memory_text_target_bank.py`，不改变训练公式或 cache 身份，不需要重生成 cache。另补正 `tests/test_macdiff_text.py` 中 scaler 测试替身的 `get_scale` 接口。真实 GPU/DDP smoke 和新版 LP 尚未执行；旧 checkpoint-130 是否最终恢复成功仍未知。

最近服务器输出：

~~~text
HEAD is now at edc924a 1
?? config/ntu60_xsub_joint/exemplar_indices.json
?? config/ntu60_xsub_joint/stage2_exemplar_seed0.json
?? vlm_pilot/
~~~

这表示 GitHub 代码已覆盖服务器手动同步的修改，上述未跟踪文件仍保留。`vlm_pilot/` 是生成文本、渲染和缓存所在目录，不是要删除的残留。这个输出确认目录保留，但没有验证其中每份缓存的完整性。

本地 HEAD 为 `edc924a`。后续检查开始时，交接与旧交接副本已经有未提交修改；本次再修改上述两个目标库文件及相关回归测试，未提交/推送，也未同步服务器。正式运行前需同步两个目标库文件；不要用 `git clean` 清理生成数据。

## 2. 环境与用户偏好

- 本地：`D:\program\MacDiff`，Windows PowerShell。
- 服务器：`/home/user9/public3/swr/MacDiff`；用户自己运行命令。本会话没有可用 SSH，不得声称替用户跑过服务器。
- 数据：从仓库根目录访问 `../data/MAMP/ntu/NTU60_XSub.npz`；训练样本 40091。
- 历史 MacDiff 训练环境：conda `macdiff`、Python 3.8、PyTorch 1.8.1+cu111、两张 RTX 4090 24GB。以服务器实际环境为准。
- Qwen 推理应使用之前能运行 Qwen3-VL 的环境；不要默认旧 macdiff 环境也能运行新版 Transformers/Qwen。
- 本地可用 Python：`C:/Users/97537/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`，有 NumPy/PIL，缺 PyTorch、Transformers、PyYAML。不能在本机宣称验证了真实模型梯度/GPU/DDP。
- 用户偏好中文、简短提示词、结构化输出、可直接复制的**单行 Linux 命令**。用户已明确授权新版文本生成和链路改动，不要每一步重新请求确认。

关键路径：

| 用途 | 路径 |
|---|---|
| 既有渲染，复用 | `vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5` |
| Qwen 模型 | `/home/user9/public3/swr/models/Qwen3-VL-8B-Instruct` |
| CLIP 模型 | `/home/user9/public3/swr/models/clip-vit-base-patch32` |
| 新版文本输出 | `vlm_pilot/ntu60_xsub_global_local_v2` |
| 新版七句 RMS cache | `vlm_pilot/ntu60_xsub_clip_sentence_cache_v3` |
| 旧 BPE local cache，保留历史实验 | `vlm_pilot/ntu60_xsub_clip_cache_v2` |

这里的 prompt v2、caption schema v2、CLIP cache v3、render v3 是不同组件的版本，不要混为同一个版本号。

## 3. 用户已经确认的文本方案

参考工作由用户指定为 2025 TMM 的 “Vision-Language Meets the Skeleton: Progressively Distillation With Cross-Modal Knowledge for 3D Action Representation Learning”，仓库 [C2VL](https://github.com/cseeyangchen/C2VL)。用户指出其公开内容主要在 text 文件夹；本项目采用的是逐样本生成的思路，不应未经核对宣称完整复现该工作的输入和生成流程。

一个原始样本可有一个或两个人；**每个人分别生成一个 global 和六个固定部位 local**。两块视频画面是同一批人的 front/side 视角，不是两个人。红色 person_index=0、蓝色 person_index=1；数量优先由 render metadata 提供。

提示词文件：

- 英文执行版：`tools/vlm_pilot/skeleton_motion_prompt_v2.txt`。
- 中文翻译：`tools/vlm_pilot/skeleton_motion_prompt_v2_zh.txt`。中文文件是翻译说明，实际要求仍生成英文描述。

已确认的核心句子：

> Global: Briefly describe the overall motion throughout the video.
>
> Local: For the head, torso, left arm, right arm, left leg, and right leg, describe each region's main motion. If a region shows no obvious movement, briefly describe its maintained pose.

对应中文：

> 全局描述：简短描述整段视频的整体运动。
>
> 局部描述：按照头部、躯干、左臂、右臂、左腿、右腿，分别描述该部位的主要运动。没有明显运动时，简短说明其保持的姿态。

每条是一句简短英文；用户明确不喜欢大量额外约束。固定部位顺序：

~~~text
head, torso, left_arm, right_arm, left_leg, right_leg
~~~

VLM 返回 `{"persons":[...]}`，脚本添加原始 `sample_index`。最终文件结构示意：

~~~json
[
{"sample_index": 0, "persons": [{"person_index": 0, "global": "...", "local": {"head": "...", "torso": "...", "left_arm": "...", "right_arm": "...", "left_leg": "...", "right_leg": "..."}}]},
{"sample_index": 1, "persons": [{"person_index": 0, "global": "...", "local": {"head": "...", "torso": "...", "left_arm": "...", "right_arm": "...", "left_leg": "...", "right_leg": "..."}}]}
]
~~~

`captions.json` 是合法 JSON 数组，每个样本一个物理行，按 sample_index 排序；JSONL 分片则每行都是独立 JSON 对象，没有外层数组，不可把两个 JSON 数组直接用 cat 拼接。

`tools/vlm_pilot/caption_output.py` 已把模型型号、设备型号、prompt、运行设置移到 `*.metadata.json`，原始回复/重试/错误等放 `*.diagnostics.jsonl`，不在每个样本文本里重复写这些信息。

## 4. 已完成的文本→缓存→训练接入

当前链路：

~~~text
双视角骨架 GIF
  → Qwen：每个样本/人物 global + 六部位句子
  → captions.json + captions.metadata.json
  → cache_clip_motion_text.py：七句分别经冻结 CLIP 编码、RMS、FP32
  → v3 NPY + manifest
  → util/person_text_cache.py：按 feeder 原始 sample/person 索引取特征
  → engine_pretrain.py / transformer_macdiff_text.py
  → Stage1 骨架 encoder
  → linear probe（仅骨架 encoder，不带文本分支）
~~~

新增/调整文件：

| 文件 | 作用 |
|---|---|
| `tools/vlm_pilot/run_caption_dual_gpu.py` | 一卡一独立 Qwen worker，分片生成、自动 resume、排序合并、进程清理与输出锁 |
| `tools/vlm_pilot/caption_output.py` | 文本主文件与运行/诊断 sidecar 分离、格式和身份校验、末行恢复 |
| `cache_clip_motion_text.py` | 新七句 CLIP 缓存生成和恢复；旧 `cache_clip_text.py` 保留 |
| `util/structured_text_cache.py` | NumPy 级格式/人物/尺度/文件身份/训练定义校验 |
| `util/person_text_cache.py` | 同时支持旧 v2 BPE 与新 v3 六句，维持原始索引和人物配对 |
| `main_pretrain.py` | v3 context/dim/归一化与 flip 保护，现有目标库接入 |
| `model/transformer_macdiff_text.py` | global/local 内容统计，去除旧诊断指标；原训练目标保留 |
| `engine_pretrain.py` | 去除 ordinary empty 骨架输出，仍过滤空人物 loss/目标更新 |
| 两份 `pretrain_madiff_text_sentence_*.yaml` | 新 cache 的固定 RMS 和逐样本目标共享组配置 |

v3 协议为 `macdiff_clip_sentence_cache_v3`：

| 文件/字段 | 含义 |
|---|---|
| `person_features.npy [N,2,512]` | 每个人的 global 句向量 |
| `token_features.npy [N,2,6,512]` | 六个完整部位句子的向量；名字保留 token，含义不再是 BPE |
| `person_valid.npy [N,2]` | 对照原始 x_train 非零人物槽位 |
| `token_mask.npy [N,2,6]` | 有该人物则六句都有效，静止姿态句也有效；空人物全部置零 |
| `samples.json` | 文本副本 |
| `manifest.json` | 定义、进度、输入/实现身份及文件校验和 |
| `context_length=7` | 训练的 global + 六部位结构容量 |
| `clip_context_length=77` | CLIP 编码一句话的 token 上限，两种长度不可混用 |

七个句子独立取 CLIP projected pooled `text_embeds`，逐向量 unit RMS 后保存 FP32。缓存里没有可训练 remap、人物/部位 embedding，也没有旧 BPE token_ids。reader 的 local positions=1..6 绑定上述部位顺序；这是结构提示，不是时间位置，也不是关节硬 attention mask。

输入需要完整覆盖原始训练行、无重复 sample_index、合法且连续的 person_index、六个精确部位。缓存生成实际流式读取 `x_train` 校验人物，不读取 `y_train`。句子超过 CLIP 上限会报错，不静默截断；model/prompt/revision 不一致不能混用。40091×两个人槽位的 global/local FP32 特征约 1.07 GiB；person0 七向量目标库约 0.535 GiB，另有元数据。

旧链路审计结论：

- 早期跨人物聚合文本配单人骨架的问题，历史 `879c232` 已改成逐人 reader/模型配对；不能靠重新写 cache 修正仍使用旧模型的计算。
- 旧 v2 是正确的 L2 CLIP 特征，能量约 1/512；训练后加 RMS 是扩散目标尺度选择，不是提取算错。新版在生成 cache 时直接完成 RMS。
- 旧 local 是 BPE 投影状态，EOS 与 global 句向量有结构性重复。新版用六个完整部位句子解决此定义问题；不能平均旧词元或重命名槽位冒充新版描述。
- 固定人物槽位、RMS、六部位顺序、padding 可在 cache 阶段固定；随机 crop 的空人物、在线 remap、历史目标更新必须保留在训练端。

## 5. 训练目标、uniformity 与最新日志定义

用户确认的逐样本模式是 `sample_target_blend`，不是旧的参数 EMA teacher 网络：

~~~text
initial_target_i = RMS(CLIP_i)
成功 optimizer step 后：
target_i = 0.9 * old_target_i + 0.1 * current_online_remap(RMS(CLIP_i))
~~~

global 和有效 local 均这样递推。T→S 用在线 remap 文本条件训练 remap；S→T 用 detach 的保存目标训练骨架 encoder 与文本噪声 decoder。**T→S 不直接给骨架 encoder 梯度是用户设计，不要作为 bug 擅自改变。** 更新使用 step 后的在线 remap；AMP 跳过 step 则不更新。不要改成每步 0.9 固定 CLIP + 0.1 remap，也不要把混合后的保存目标再次强行 RMS，那都会改变公式。

`shared_memory` 后端在同机 RAM 保存目标，只在 checkpoint 时写兼容 SQLite 快照；旧 SQLite 后端可回退。恢复必须有同次保存、同组的 `checkpoint-X.pth` 和 `checkpoint-X-target-bank.sqlite`。`current.shared.json` 不是可恢复的完整目标库。旧 v2 目标库约 1.6 GiB；新版 person0 七句约 0.535 GiB。共享内存的 Linux /dev/shm、GPU性能和真实 DDP 尚无本会话验证；不能保证消除用户之前反馈的每轮多约10分钟。

用户基于“remap 后方差低/坍缩”询问 uni 是否合适，最终决定**暂时保留原 uniformity，只改日志**：

- 原 `masked_text_uniformity_loss` 仅对每个样本内有效 local 两两平方 cosine 做均值，含对角项；不含 global、不跨样本。
- 六个 local 的对角下限约 1/6，旧约20个词元则约 1/20。它可抑制句内所有 local 相同，但全体样本共用相同的六个正交向量也能达到下限，因此不能声称解决了跨样本坍缩。
- 按部位跨样本方差下限/VICReg式正则只是讨论，未实施。用户没有要求新增这项 loss。

最新输出改动已经提交在 `edc924a`：

| 指标/行为 | 最新定义 |
|---|---|
| 空骨架普通提示、`empty_skeleton_persons` | 已取消输出和记录；内部过滤仍执行 |
| `text_target_drift_mse` | 已删除计算和日志 |
| `text_batch_variance` | 在线 remap 后 global 与所有有效 local 内容向量拼接，逐通道总体方差 `var(unbiased=False)` 后取均值 |
| fixed_clip 的 variance | 该模式没有 remap，统计固定内容向量 |
| `text_energy` | 仍统计保存目标的 global 能量，没有随 variance 一起改群体 |

variance 不含人物/部位 embedding、padding、非活动空人物；每卡当前 microbatch 计算，日志沿用原有各卡/各步汇总，不是两卡合并后的总体方差。它包含部位之间的差异，不能单独证明同部位跨样本健康。历史日志的 variance 是保存 global 目标的批内方差，不能与新版直接比较。取消 empty 输出不意味着允许空 crop 参与 loss 或借 person1 替代 person0；全批为空仍可报错。

## 6. 下一步计划与可直接执行的服务器命令

顺序：确认新版文本完整 → 抽检六部位内容 → 编码并验证 v3 cache → 固定 RMS 双卡 smoke → 新目录训练 → 同协议 LP → 再考虑其他优化。各命令均从服务器仓库根目录执行；Qwen、CLIP 提取、旧训练可能需要各自原先可用的环境。

### 6.1 双卡生成新版文本并自动恢复

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/vlm_pilot/run_caption_dual_gpu.py --rendered_root vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5 --model /home/user9/public3/swr/models/Qwen3-VL-8B-Instruct --output_dir vlm_pilot/ntu60_xsub_global_local_v2 --gpus 0,1 --expected_samples 40091
~~~

GPU0 偶数、GPU1 奇数；一卡一份模型，不是模型切两卡。launcher 自动给 worker 添加 `--resume`，**中断后原样重跑同一条命令即可**。可先加 `--max_samples 4` 试跑，再去掉继续全量；`--dry_run` 只检查待生成数量与命令，`--merge_only` 仅合并已存分片。launcher 参数是下划线，不要写成 cache 的连字符参数。

输出目录需保留：

- `captions.json`、`captions.metadata.json`。
- `shard0.jsonl` / `shard1.jsonl` 和各自 `*.metadata.json`。
- 各自 `*.diagnostics.jsonl`、`*.log`。

resume 跳过合法 accepted，重试失败/无效样本；模型、提示词、revision 和渲染来源需一致。只对未完整写入的最终 JSONL 行做备份并移除；完整坏行和身份不符会报错，不能强行忽略。输出锁阻止同目录双启动；Ctrl+C 会停止子进程并尽量生成排序后的部分结果。**最终 JSON 存在不等于全量完成。**

检查完成摘要（应 expected=accepted=40091、missing_count=0、complete=true）：

~~~bash
cd /home/user9/public3/swr/MacDiff && python -c "import json; print(json.load(open('vlm_pilot/ntu60_xsub_global_local_v2/captions.metadata.json'))['summary'])"
~~~

随后抽检单/双人、左右部位、静止姿态、brief 程度与视频配对。`tools/vlm_pilot/inspect_random_captions.py` 可帮助检查；先查看其 CLI，不杜撰参数。

### 6.2 把新文本编码为可复用 CLIP cache

使用之前能成功提取 CLIP 的环境：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u cache_clip_motion_text.py --data-path ../data/MAMP/ntu/NTU60_XSub.npz --captions vlm_pilot/ntu60_xsub_global_local_v2/captions.json --clip-model /home/user9/public3/swr/models/clip-vit-base-patch32 --output-dir vlm_pilot/ntu60_xsub_clip_sentence_cache_v3 --batch-size 16 --resume
~~~

一个 batch 是16个人、112句。单卡即可；CLIP 编码完成后，后续训练直接读取 NPY，不重复加载/运行 CLIP。脚本必读配套 `captions.metadata.json`，不能只复制文本主文件。

cache 的 `--resume` 需显式写；重复上述命令恢复。不完整缓存先 flush 每人的七向量再发布 `completed_persons`；已完整缓存经校验后直接复用。dataset、captions、metadata、CLIP 或相关实现 SHA256 变更必须换输出目录，不要删除 manifest 绕过身份保护。暂停缓存生成时不要继续修改其输入文本/metadata。

检查进度：

~~~bash
cd /home/user9/public3/swr/MacDiff && python -c "import json; m=json.load(open('vlm_pilot/ntu60_xsub_clip_sentence_cache_v3/manifest.json')); print({k:m.get(k) for k in ('protocol','complete','sample_count','completed_persons','total_persons','context_length','clip_context_length')})"
~~~

应 complete=true、sample_count=40091、completed_persons=total_persons。total_persons 是实际人物数量，不一定等于40091或80182。若出现人物槽位不匹配，定位 sample/person，检查渲染/描述；32帧采样可能漏掉短暂出现的人，不能用另一人的文字补齐。

### 6.3 固定 RMS 双卡 smoke，再从头训练

两份新配置均 `context_length=7`、`flip=False`、文本 decoder 隐藏256、400轮：

| 配置 | 训练模式 | T→S / S→T | 文本 uni | 共享骨架 decoder |
|---|---|---|---|---|
| `pretrain_madiff_text_sentence_fixed_rms.yaml` | fixed_clip / rms | 0 / 0.1 | 0 | False |
| `pretrain_madiff_text_sentence_sample_target_blend_shared.yaml` | sample_target_blend / rms | 0.1 / 0.1 | 0.02 | True |

先在独立目录运行：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10252 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml --batch_size 2 --accum_iter 1 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --max_train_steps 2 --num_workers 0 --output_dir output_dir/ntu60_xsub_sentence_fixed_smoke --log_dir output_dir/ntu60_xsub_sentence_fixed_smoke/tensorboard
~~~

首次真实运行保留完整 cache validation；通过后才考虑 `--skip_text_cache_validation`，其 header/size 模式不能发现同大小内容篡改。上面 smoke 不是 epoch 性能测量，也尚未执行成功。

正式固定组：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10252 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml --batch_size 32 --accum_iter 2
~~~

正式逐样本共享组；先确认 /dev/shm 和快照磁盘空间：

用户确认 cache 完成后的检查继续采用此组。该组的两卡 smoke 应覆盖共享目标库和两个 optimizer step；在原训练环境中运行，使用新的 smoke 输出目录：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10254 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml --batch_size 2 --accum_iter 2 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --max_train_steps 4 --num_workers 0 --output_dir output_dir/ntu60_xsub_sentence_sampletarget_shared_smoke_20261001 --log_dir output_dir/ntu60_xsub_sentence_sampletarget_shared_smoke_20261001/tensorboard
~~~

通过后，在配置的正式目录从头训练：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10254 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml --batch_size 32 --accum_iter 2
~~~

两组使用相同GPU，应顺序运行；共享组正式训练前也需要真实 smoke。两卡每卡32×累积2=有效 batch128，CLI 覆盖 YAML 中的 batch64/累积1。新配置各自已有独立输出目录，**不要添加旧实验的 --resume**。完整 resume 只用于同文本、同配置、同输出实验；新 sample_target 实验需要该实验自己的配对目标库快照。

这两组同时改变了目标模式/共享设置，不是仅文本版本的单变量对照。比较新旧文本应优先对齐固定 RMS 0.1 的其余预训练与 LP 参数。

新固定组完成后，LP 示例（只加载骨架 encoder）：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10253 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_fixed_rms_st01/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_fixed_rms_st01_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_sentence_fixed_rms_st01_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~

该 LP 有效 batch128。历史85.82%基线的 LP 有效 batch64，不能直接称严格同协议提升；需以统一 LP 设置对照。共享组 LP 需换成它自己的 checkpoint 和独立输出目录。

本次推荐共享组 LP 使用每卡32、累积1，即有效 batch64，与历史85.82%基线交接中记载的 batch 一致。未找到该85.82%实验完整原始 CLI/日志，不能据此声称核实了所有历史参数：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10255 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_lp_399_bs32 --log_dir output_dir/ntu60_xsub_macdiff_sentence_sampletarget01_shared_lp_399_bs32/tensorboard --batch_size 32 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~

## 7. 已做验证与证据边界

cache 完成后的本次复查实际重跑 `tests.test_structured_text_cache`、`tests.test_clip_text_cache`、`tests.test_sample_text_target_bank`、`tests.test_shared_memory_text_target_bank`、`tests.test_macdiff_text`：63项，33项通过、30项因缺 Torch/Transformers/PyYAML 跳过，无失败。其中8项新增 CPU 回归覆盖两个目标库后端的孤立临时/完成快照重建、已配对 checkpoint 拒绝覆盖、发布失败保留旧完整库和发布阶段出现模型文件时拒绝覆盖。Python3.8 AST、smoke/LP CLI parser 和 diff whitespace 检查通过。此次没有服务器执行能力，GPU/AMP/DDP 未实测。

以下是前面开发阶段的结果，不是本次文档更新重跑出来的：

- caption 输出与双卡调度：24项 CPU/模拟测试通过；不等于真实 Qwen 双卡推理成功。
- v3 cache/旧 reader/目标库等某轮回归：63项，33项 CPU 通过、30项因缺 Torch/Transformers/PyYAML 跳过。
- 最近日志修改那轮：55项，25项 CPU 通过、30项依赖测试跳过。覆盖 `tests.test_macdiff_text`、`tests.test_sample_text_target_bank`、`tests.test_shared_memory_text_target_bank`、`tests.test_structured_text_cache`、`tests.test_clip_text_cache`。
- global+local variance 用 NumPy-backed torch API 模拟验证：global [[0,2],[2,0]]、有效 local [[4,0],[0,4]]，合并 variance=2.75；无效 padding/NaN 被排除。不是实际 Torch 训练测试。
- Python3.8 AST 语法检查、CLI help、Git diff whitespace 检查通过；不是实际 Python3.8/GPU执行。

相关测试文件包括 `tests/test_caption_output.py`、`tests/test_caption_dual_gpu.py`、`tests/test_structured_text_cache.py`、`tests/test_macdiff_text.py`、`tests/test_shared_memory_text_target_bank.py`。以后若修具体失败，再运行对应检查；文档编辑不必反复运行缺依赖的大套件。

尚未验证：真实 Transformers CLIP 编码、服务器全量新版数据校验、Linux 共享内存、真实模型前向/梯度、AMP/DDP、性能改进、新版 LP。用户提供日志后再补状态。

## 8. 绝对不要再踩的坑与仍未解决的限制

1. **不要对服务器执行 git clean -fd/-fdx。** `vlm_pilot/` 未被 Git 跟踪且没有整体忽略，其中有重要生成结果；清理可能删除文本/cache。本次 git reset --hard 仅因用户明确要求以 GitHub 覆盖代码而执行，已成功，不能把这个授权套到未来任意修改。命令针对服务器，不要误在 Windows 本地执行覆盖。
2. Git reset --hard 会覆盖跟踪改动及挡住远端跟踪路径的未跟踪文件/目录；其他未跟踪数据通常保留。`.gitignore` 不是发生路径冲突时的绝对保护。最近两个 exemplar JSON 和 vlm_pilot 已保留，不要为获得空 git status 而删除。
3. 新旧文本/cache/实验目录分开；保护旧 v2 数据，但用户已授权生成新版。不能用旧“禁止重生成”阻断现在的新任务，也不能覆盖旧实验来省事。
4. captions.metadata.json 必须与 captions.json 一起保存；生成 resume 还要保留分片及其 metadata。只剩最终 JSON，不能期待 launcher 自动恢复全部原会话身份/分片。
5. 保持原始 sample_index 与 person_index；person0 配自己的文本，空 person0 不替换为person1。front/side 是视角，不是人物。shuffle/crop 后 feeder 原始索引仍是查表依据。
6. cache 结构长度7不是 CLIP长度77；local 是六句完整语义，不是六个词/BPE，也不是按时间切的六段动作。
7. 左右语义需抽检。现有渲染没有明确逐关节左右标注；格式合法不等于识别正确。v3 当前禁止 flip=True，因为训练没有相应文本交换/方向转换机制。
8. 全段描述与 p_interval=[0.5,1] 的随机时间 crop 可能不一致；某个主要动作不一定在当次 crop 可见。完整序列或带时间边界描述的消融未实施。
9. 渲染逐帧减去person0 root，世界坐标位移已丢失；global 不能补出没有展示的信息。已有 render provenance 的路径/size/mtime 也不能完全证明历史 captions 来自内容相同的NPZ；新 cache 仅能绑定当前 NPZ SHA256。
10. S→T 统一平均 global/local。旧约20个local时 global约占1/21，新六句时约1/7；外层权重相同并不意味着内部监督分配相同。当前没有新增 global/local 独立权重。
11. text_batch_variance 是在线global+local、text_energy 是保存global目标；旧 variance/新 variance、L2/RMS、旧20词元uni/新6句uni不能直接跨口径解释。
12. 保留原uni是用户决定；不要把它宣传成跨样本抗坍缩正则，也不要擅自加新正则/辅助loss。RMS能量≈1也不代表方向不坍缩或语义好。
13. 新文本不可完整 resume 旧 checkpoint/旧目标库，涉及缓存身份、结构 embedding 尺寸及历史目标改变。不同 share设置、旧参数EMA和sample_target也不能互相完整resume。不要用 strict=False 部分加载却称完整恢复。
14. sample_target 续训必须模型与同名目标库快照配对；不要缺库时静默初始化，不删除配对快照省空间。LP只需模型，不需要目标bank。快照是GiB级，长期保存需关注空间。
15. 历史恢复命令曾把训练后直接拼 `cd ... && LP`，缺少训练与cd间分隔，argparse报 unrecognized arguments: cd；那次根本没有恢复模型/目标库。训练后自动LP应是 `完整训练命令 && 完整LP命令`。只有父进程 CalledProcessError 时，要找前面的真实rank traceback。
16. share=True 共享的是**原生骨架重建与T→S骨架decoder主体**，不是T→S和S→T共用decoder。S→T独立文本decoder隐藏256/输出512仍有约0.5的各向同性噪声期望MSE下限；不是有限batch硬下限，也不是LP上限。
17. text_decoder_hidden_dim=512、noisy-input skip、辅助干净global预测等仅讨论未实施。扩宽独立文本decoder可以继续share骨架decoder，但旧256 checkpoint不能完整恢复为512。T→S 512→256条件投影没有相同的512噪声输出下限；是否限制语义监督尚无证据。
18. loss下降、shuffle效应、目标方差都不能代替LP。两条文本方向日志是未乘外层权重的原始MSE，不能把loss占比当作encoder梯度占比。
19. 真实 LP 读出是 `ActionHeadLinprobe2` 的25关节×256=6400维（人物/时间平均），随后BN+Linear；`feature_only=True` 的global256不是实际LP输入。比较需best对best且相同训练协议。
20. 旧诊断脚本各有模式限制：`diagnose_ema_bidirectional.py` 针对旧参数EMA，`compare_fixed_clip_conditions.py` 针对固定CLIP，`diagnose_text_conditioning.py` 针对旧remap。不能未经适配强读sample_target或v3 checkpoint并解释为新方案结果。

## 9. 历史实验背景，不能当作新版结果

旧 Qwen 单段逐人物描述的40091条已全量完成，历史18个GIF错误已补齐；旧v2 cache可用。**这是旧版，不证明global+六部位新版已完成。**

| 历史方案 | 用户提供的LP best |
|---|---:|
| 原始MacDiff | 约85.86% |
| 第一版双向remap/动态目标，S→T=1 | 83.03% |
| 固定CLIP原尺度，S→T=1 / 0.1 | 84.64% / 85.95% |
| 固定CLIP RMS，S→T=1 / 0.1 | 约83.7% / 85.82%（暂定基线） |
| 旧参数EMA双向，不share / share | 85.77% / 85.78% |
| 逐样本0.9/0.1保存目标 | 有早期训练日志，尚无已确认LP |
| 新global+六部位文本/v3 cache | 尚无已确认预训练或LP结果 |

旧参数EMA和逐样本目标不要混同。旧EMA诊断global跨样本明显集中，而local未同样收缩；shuffle后S→T总MSE增幅很小，但约0.5结构下限稀释百分比，不能直接断言骨架条件没用。完整诊断、数字与历史命令见旧交接副本。

最近明确运行的旧sample_target是共享骨架decoder组，用户曾尝试从 `output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared/checkpoint-130.pth` 恢复；第一次因上述shell拼接失败，修正后成功与否未知。若新会话用户继续这个旧任务，先确认配对 `checkpoint-130-target-bank.sqlite` 与真实恢复日志，不能因为新版准备工作就擅自中断或改旧训练。

旧样本目标epoch0～19日志：保存global方差约0.13008→0.04038、能量1→0.79619、drift0→0.73816；总loss1.265→0.08744、S→T约0.535。目标有集中趋势但没有LP结论。这些日志使用旧方差定义，且旧drift已从新日志删除。

历史几何文件 `handoff_artifacts/stage1_text_geometry_summary.json` 是固定均衡train批的文字/骨架几何比较，不是测试准确率；其文字是两人聚合global，与当前person0/七句协议不同。不要整份dump或编造新版增益。

## 10. 新会话建议读取顺序

1. 本文件第1、5、6、8节：当前状态、用户决定、可执行步骤与坑。
2. `tools/vlm_pilot/TEXT_CACHE_CHAIN_AUDIT.md`：链路审计及新版缓存细节。
3. 实际执行的提示词、launcher、cache脚本与选定sentence YAML。
4. 用户提供的服务器文件/日志：确认生成summary、cache manifest、smoke traceback或训练配置。
5. 需要继续旧实验时才读 `handoff_artifacts/handoff_before_20261001_refresh.md`，以及 `tools/vlm_pilot/STAGE1_TEXT_DIFFUSION.md`、`tools/vlm_pilot/STAGE1_TEXT_GEOMETRY.md`、根目录 `TEXT_CONDITION_DIAGNOSTIC.md` / `STAGE1_READOUT_COMPARISON.md`。

收到新消息时优先承接用户实际运行到的步骤；没有新版完成证据就先验证，看到真实报错再修具体问题。不要重复要求用户手动同步已在edc924a中的代码，不要把未实施建议当作已完成工作。
