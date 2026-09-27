# MacDiff Stage1 文本扩散交接（2026-09-27）

本文写给完全没有上下文的新会话。请先读“当前状态”和“绝对不要再踩的坑”，再看代码和命令。用户偏好中文、可直接复制的单行 Linux 命令。历史 handoff_stage2_legacy_20260905.md、handoff_vlm_pilot_legacy.md 和旧实验脚本仅供追溯。

## 1. 当前状态：任务、完成情况、卡点

任务：在 NTU60 XSub 的 MacDiff Stage1 骨架预训练中加入由骨架描述得到的文本监督，尝试改善骨架 encoder 的动作表示和下游 linear probe（LP）。不是 Stage2/OSE，也不是文本生成。LP 只加载骨架 encoder，文本分支不进入下游模型。

用户最后明确的新设计是：每个原始训练样本保存自己的骨架→文本去噪目标；初始目标是固定 CLIP 向量的逐向量 RMS 归一化结果。该样本经过一次成功的 optimizer step 后，用当时在线 remap 网络的输出更新：

    target_i ← 0.9 × old_target_i + 0.1 × current_online_remap(CLIP_i)

global 文本向量和有效 local token 向量都这样更新。这里固定的是更新比例，不需要对 remap 网络参数做 EMA。这个递推在数学上属于“目标值的 EMA”，但与之前的“EMA teacher 网络”不是同一个实验。alpha=0.1 时旧目标贡献约每 6.58 次该样本更新减半；初始 CLIP 不会被永久锚定。如果用户将来要永久混合 0.9 固定 CLIP + 0.1 当前 remap，那是另一个无历史状态的公式，不能擅自替换当前确认过的公式。

新模式 sample_target_blend 已在本地实现；独立/共享骨架 decoder 两份配置也已写好。本地纯 NumPy/SQLite 测试和语法检查通过。本机没有 PyTorch，未完成新模式的真实模型前向、GPU、AMP 或双卡测试。服务器没有本地可用 SSH；没有证据表明新模式已同步、启动、训练或测过 LP。当前实际卡点是用户把本地改动同步到服务器，并在服务器验证和运行。绝不能把旧 EMA 网络实验的 LP 或诊断结果说成新逐样本方案的结果。

## 2. 环境、数据和历史结果

本地工作区：D:/program/MacDiff，Windows PowerShell。服务器根目录：/home/user9/public3/swr/MacDiff；历史环境为 conda macdiff、Python 3.8、PyTorch 1.8.1+cu111、两张 RTX 4090 24GB。用户自己同步代码和运行。数据：../data/MAMP/ntu/NTU60_XSub.npz。现成 v2 CLIP 缓存：vlm_pilot/ntu60_xsub_clip_cache_v2。不要重新生成描述或缓存。

Qwen3-VL-8B 已为 NTU60 XSub train 的 40091 个样本生成逐人物英文骨架描述，历史 18 个 GIF 错误已补齐，accepted_unique=40091、missing=[]。CLIP global/local token 缓存已完成验证。当前只用 skeleton person0 配它自己的文本；空 person0 裁剪应从 loss 和目标更新排除，不得改用 person1。local token 按 batch 动态 padding，平均约 20 个有效 token，并非固定 76。双视图 GIF 的 front/side 不是两个人。

历史 LP best（均为用户提供的单次数字，按其要求先比较数值；协议差异仍需记住）：

| 方案 | LP best |
|---|---:|
| 原始 MacDiff | 历史约 85.86% |
| 第一版双向 remap、动态目标、S→T 权重 1.0 | 83.03% |
| 固定 CLIP 原尺度、S→T 权重 1.0 | 84.64% |
| 固定 CLIP 原尺度、S→T 权重 0.1 | 85.95% |
| 固定 CLIP RMS、S→T 权重 1.0 | 约 83.7% |
| 固定 CLIP RMS、S→T 权重 0.1 | 85.82%，用户暂定基线 |
| 旧网络参数 EMA 双向，不 share decoder | 85.77% |
| 旧网络参数 EMA 双向，share decoder | 85.78% |
| 当前逐样本 0.9/0.1 目标方案 | 尚未训练、尚无 LP |

新旧 LP 的训练 batch 设置可能不同。旧固定 CLIP RMS 85.82% 的 LP 有效 batch 是 64，最近两组参数 EMA LP 有效 batch 是 128；两组参数 EMA 彼此可比，与 85.82% 仅作历史数值对照。LP 要比较 best-vs-best，不能混用 best 和末轮。

## 3. 旧参数 EMA 实验及刚跑完的诊断：结论和边界

旧实验配置为 config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml 及 _shared.yaml，text_target_mode=ema_remap、text_target_momentum=0.999。它训练一个在线残差 remap，同时维护一个不反传的 EMA teacher remap；S→T 去噪目标来自 teacher 网络输出。用户指出这并非他本意。旧实验已经完成 epoch 399 及 LP，不要再当作“待运行方案”。

针对旧实验的新诊断脚本：diagnose_ema_bidirectional.py；测试：tests/test_ema_bidirectional_diagnostic.py。用户在服务器跑出的完整 JSON 在 C:/Users/97537/.codex/attachments/d027fa4a-f322-41be-9472-8ebf4c422cfa/已粘贴的文本.txt，协议 ema_bidirectional_targeted_diagnostic_v1。诊断使用 256 个固定训练 crop；条件交换的成对分析可用 254 个样本，2 个不能配对。同一比较内固定 crop、mask、噪声、t；t=100/500/900。此脚本和结果只针对旧参数 EMA checkpoint，不能强行载入当前 sample_target_blend checkpoint。

| 诊断量（epoch 399） | share | 不 share | 固定 CLIP |
|---|---:|---:|---:|
| global teacher 平均通道方差 | 0.03298 | 0.00822 | 0.13347 |
| global teacher 跨样本平均余弦 | 0.96689 | 0.99175 | 约 0.866 |
| local teacher 平均通道方差 | 0.97584 | 0.97662 | 0.53532 |
| global teacher 与对应 CLIP 的平均余弦 | 0.17223 | 0.07177 | — |
| global teacher 与 CLIP 的中心化两两余弦关系 Pearson | 0.80189 | 0.80459 | — |

解释：旧参数 EMA 的 global 目标明显集中，尤其是不 share；但 local token 方差并未一起收缩。中心化的样本间关系仍有约 0.80 的相关，不能说全部语义结构消失。share 的 global 方差约为不 share 的 4 倍，但 LP 仅高 0.01 个百分点，不能推出方差越大 LP 越好。

把正确配对条件打乱后，去噪 MSE 相对增幅：

| 方向与交换 | share：t=100/500/900 | 不 share：t=100/500/900 |
|---|---|---|
| 文本→骨架：打乱全部文本条件 | 4.19% / 6.88% / 8.33% | 4.01% / 7.44% / 9.13% |
| 文本→骨架：只打乱 global | 0.08% / 0.11% / 0.22% | 0.08% / 0.14% / 0.18% |
| 骨架→文本：打乱全部骨架条件 | 0.38% / 0.28% / 0.13% | 0.49% / 0.29% / 0.13% |

文本→骨架确实利用了配对信息，主要来自 local token；骨架→文本有正的配对依赖但幅度很小。打乱后的损失上升只证明任务使用配对条件，不证明动作类别语义质量，也不证明 LP 改善。固定 CLIP RMS 权重 1.0 曾有较强条件效应但 LP 约 83.7%，就是反例。旧参数 EMA 的 share / 不 share LP 85.78% / 85.77%，与固定 RMS 0.1 的 85.82% 接近，未显示稳定增益。

用户曾贴出 share 组 0～399 epoch 训练日志：总 loss 从 1.265 降到约 0.0696，text_target_drift_mse 从 0.011 升到约 1.66，S→T 原始 MSE 后期约 0.518，T→S 约 0.021。不要把 loss 下降当作 LP 提升；两条文本扩散日志都是乘 0.1 权重之前的原始 MSE。

## 4. 当前 sample_target_blend 实现：必须按此理解

代码：

- model/transformer_macdiff_text.py：新增 text_target_mode=sample_target_blend，参数 text_target_update_ratio=0.1；只有在线 remap，没有 text_target_remap teacher 网络。T→S 用在线 remap 后 global/local 文本作条件；S→T 从目标库接收 detach 的 global/local 去噪目标，反传到骨架 encoder 和文本噪声 decoder，不直接反传到 remap。
- util/sample_text_target_bank.py：以原始样本索引为键的持久 SQLite 目标库。首次访问从固定 RMS CLIP global/local 初始化。每次成功 optimizer step 后，用更新后的在线 remap 输出按 0.9/0.1 更新当步样本目标。两卡本地 DDP 通过 SQLite 事务共享同一库。空 person0 / 无有效 token 的非活动行返回零目标并跳过更新。
- engine_pretrain.py：在 forward 前取该样本上次目标，按累积窗口收集非空 person0 索引；成功 optimizer step 后更新目标。AMP 跳过 step 时不更新。
- main_pretrain.py：准备目标库、记日志 target_bank_initialized_samples，给每份 checkpoint 建对应目标库快照。比如 checkpoint-399.pth 对应 checkpoint-399-target-bank.sqlite。resume 先载入同组模型 checkpoint，再从同名快照恢复目标库；缺快照应报错，不可偷偷重置。
- util/misc.py：resume 检查新模式、0.1 更新比例、uniformity、共享设置和缓存身份。
- 两份新配置：config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml（独立 decoder）和 pretrain_madiff_text_rms_sample_target_blend_shared.yaml（共享 decoder）。均为 RMS、T→S 0.1、S→T 0.1、文本 uniformity 0.02、400 epoch，差别仅共享开关和输出目录。

新模式仍是 512 维 RMS CLIP 输入、随机初始化的可训练残差 remap 和 512 维文本去噪目标。文本 person/position embedding 只加到 T→S 条件，不写入 S→T 干净目标。混合后的目标不再保证每个时刻 text_energy 恰好为 1，因为两个 RMS 向量的加权平均可降低能量；首次目标应约为 1，后续以实际日志为准。text_target_drift_mse 是目标相对固定 RMS CLIP global 向量的 MSE，不能要求它一直近零。

SQLite 按 float32 存 global 加有效 local token，约 40091 个样本、平均 20 local token；当前库及每个 checkpoint 快照可能达到 GiB 级。40 多份快照会占用较多磁盘；服务器运行前检查空间。LP 只读模型 checkpoint，不需要目标库快照；继续预训练必须保留匹配快照。旧参数 EMA checkpoint 不能用于新模式 resume；独立与共享组也不能互相 resume。不要删除已有库或改写旧实验目录。

## 5. 下一步和准确命令

先同步下面 7 个本地文件到服务器相同相对路径：model/transformer_macdiff_text.py、engine_pretrain.py、main_pretrain.py、util/misc.py、util/sample_text_target_bank.py、两份 pretrain_madiff_text_rms_sample_target_blend YAML。旧诊断脚本和测试若需要复查也可同步，但不参与正式训练。服务器首次启动建议先在独立的 smoke 输出目录跑两步以验证 PyTorch 1.8.1、AMP、SQLite、双卡及 checkpoint 快照；不要把 smoke 输出目录用于正式训练。拿到真实 traceback 再修具体问题。本地无服务器执行能力。

Smoke（独立 decoder，两卡，每卡 batch 32，累积 2 次，2 个 micro-step，独立输出目录）：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10246 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml --batch_size 32 --accum_iter 2 --epochs 1 --warmup_epochs 0 --min_lr_epochs 0 --max_train_steps 2 --output_dir output_dir/ntu60_xsub_sampletarget01_smoke --log_dir output_dir/ntu60_xsub_sampletarget01_smoke/tensorboard --skip_text_cache_validation
~~~

正式预训练使用两卡、每卡 batch 32、accum_iter 2，有效 batch 128。两个任务使用同一对 GPU，应顺序运行。全新目录从头开始，不加 --resume。

独立 decoder：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10242 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml --batch_size 32 --accum_iter 2 --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01/tensorboard --skip_text_cache_validation
~~~

共享 decoder：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10244 main_pretrain.py --config config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml --batch_size 32 --accum_iter 2 --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared/tensorboard --skip_text_cache_validation
~~~

两组完成 epoch 399 后，分别用相同 LP 设置：两卡每卡 batch 64、100 epochs、lr 0.1、seed 0、dist_eval。LP 不需要目标库 SQLite。

独立 decoder 的 LP：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10243 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~

共享 decoder 的 LP：

~~~bash
cd /home/user9/public3/swr/MacDiff && OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10245 main_linprobe.py --config config/ntu60_xsub_joint/linprobe_madiff.yaml --finetune output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared/checkpoint-399.pth --output_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared_lp_399_bs64 --log_dir output_dir/ntu60_xsub_macdiff_rms_st01_sampletarget01_t2s01_shared_lp_399_bs64/tensorboard --batch_size 64 --accum_iter 1 --epochs 100 --lr 0.1 --seed 0 --dist_eval
~~~

用户若提供新模式训练日志，先核对 text_target_mode=sample_target_blend、text_target_update_ratio=0.1、rms、share 开关、分支权重、目标库初始化数、checkpoint 与目标库快照是否配对，再分析 loss 和 LP。若只给 torch.distributed.launch 的 CalledProcessError，要找其前面的真实 rank traceback。若做严格历史基线比较，需要用相同 LP 命令参数重跑固定 RMS 0.1 基线，因为旧 85.82% 是另一 LP batch。

## 6. 本地验证、工作区和绝对不要再踩的坑

本地新增 tests/test_sample_text_target_bank.py 覆盖逐样本 0.9/0.1 递推、独立样本历史、快照恢复、空 person 行。使用 Codex 捆绑 Python 的 unittest tests.test_sample_text_target_bank tests.test_ema_bidirectional_diagnostic：6 项中 3 项纯 NumPy/SQLite 通过，3 项 PyTorch 测试因本机无 torch 跳过；py_compile 通过；git diff --check 通过，仅有现存 CRLF 提示。没有服务器 GPU/AMP/DDP 运行结果。不要将“通过本地测试”说成生产双卡已验证。

当前 git 工作区有未提交改动：engine_pretrain.py、handoff.md、main_pretrain.py、model/transformer_macdiff_text.py、util/misc.py；未跟踪：两份 sample_target_blend YAML、util/sample_text_target_bank.py、tests/test_sample_text_target_bank.py、diagnose_ema_bidirectional.py、tests/test_ema_bidirectional_diagnostic.py。不要 reset、clean、覆盖或声称已 commit；handoff.md 本来就有用户先前修改，本次按用户要求更新。没有生成 zip。

必须记住：

1. 当前每个样本的递推目标是用户明确选定的方案。不要再把“EMA 网络”当作用户意图，也不要把旧参数 EMA teacher 的 checkpoint、诊断、LP 混入新方案。固定比例作用于保存的目标值；在线 remap 参数正常梯度训练。
2. 不从旧 EMA、第一版 remap、固定 CLIP 或另一 share 设置 resume 新模式。新模式 resume 必须同组同配置、模型 checkpoint 与 -target-bank.sqlite 配对；不得只恢复模型而重新初始化目标库。
3. 不重新生成 40091 条描述或 v2 CLIP 缓存。现有缓存可信时可用 --skip_text_cache_validation；完整 hash 和逐行扫描耗时且双 rank 都执行。
4. 骨架 person0 必须配自己的文本；空 person0 不用 person1 代替。目标库应跳过非活动空行。front/side 是同一人的不同视角。
5. 原尺度 L2 CLIP 每通道能量约 1/512，RMS 约 1。不能直接跨尺度比较 S→T MSE、text_energy 或绝对方差。text_energy≈1 只说明尺度，不能证明语义质量；新方案混合后还可能偏离 1。
6. T→S 损失下降不代表骨架 encoder 一定改善；T→S 不直接更新 encoder。S→T 打乱测试增幅不等于 LP 改善。要看实际 LP best。
7. 总 loss 包含多个加权项，不能和原始 MacDiff native loss 直接比。同口径 native 是 loss_diff + 0.02*loss_uniformity；文本两方向日志 MSE 未乘 0.1。
8. 预训练 --batch_size 是每卡值，32×累积2×两卡=有效128。最近 LP 每卡64×两卡=有效128；历史 RMS0.1 LP 有效64。共享/独立对照必须保持相同 LP 设置。
9. checkpoint-0 已完成第一轮，不是随机初始化。训练 log.txt 可能追加重启段，出现重复 epoch；分析时按重启分段或取每个 epoch 的最后一条。
10. OSE 默认参数出现在 args 不代表启用；看 enable_ose=False。torch.distributed.launch 的 CalledProcessError 多半只是父进程摘要。误按 Ctrl+Z 会暂停训练进程却占着端口，先查进程再重启。
11. 不用只支持旧 remap 的 diagnose_text_conditioning.py、只支持固定 CLIP 的 compare_fixed_clip_conditions.py，或只支持旧参数 EMA 的 diagnose_ema_bidirectional.py 强行读取新模式 checkpoint。需要先适配目标构造及 checkpoint 加载。
12. 服务器空间需容纳 SQLite 当前库和 checkpoint 快照。不要为省空间擅自删匹配快照；那会使相应模型 checkpoint 无法完整续训。
